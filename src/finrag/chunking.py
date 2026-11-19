"""Split sections into retrieval chunks with citation metadata.

- Prose is packed sentence by sentence up to `max_words`, never across a section boundary,
  with `overlap_words` of trailing context carried into the next chunk.
- Each table becomes its own chunk (or several, split by rows, with the header rows repeated
  in each piece) so a row label never gets separated from its column headers.
- Every chunk gets a `search_text` with a short context header (company, form, fiscal period,
  section, units) in front of the body. BM25 and the embedding both use it, so a passage
  saying "revenue increased 16%" can still match "Microsoft FY2024 revenue growth".
  Citations display the bare `text`.

Sizes are in words, as a cheap proxy for tokens. The embedding stage logs how many chunks
go over the model's token limit, so this proxy gets checked.
"""

import logging
import re
from dataclasses import asdict, dataclass

from finrag.ingest.models import FilingRecord
from finrag.parsing.html_blocks import Block
from finrag.parsing.sections import Section
from finrag.parsing.tables import detect_units, leading_header_rows, render_rows
from finrag.parsing.text_utils import word_count

logger = logging.getLogger(__name__)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")
MAX_TITLE_LENGTH = 200
UNITS_LOOKBACK_BLOCKS = 3


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 250
    overlap_words: int = 40
    table_max_words: int = 220
    min_words: int = 8


@dataclass(frozen=True)
class FilingContext:
    """Filing-level metadata copied onto every chunk (fiscal labels already finalized)."""

    record: FilingRecord
    fiscal_year: int | None
    fiscal_period: str | None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    accession: str
    ticker: str
    cik: int
    company_name: str
    form: str
    is_amendment: bool
    fiscal_year: int | None
    fiscal_period: str | None
    fiscal_year_end: str | None
    period_of_report: str | None
    filing_date: str
    part: str
    item: str
    section_title: str
    chunk_type: str
    units: str | None
    seq: int
    text: str
    search_text: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DraftChunk:
    """Chunk body before filing metadata and sequence numbers are attached."""

    section: Section
    chunk_type: str
    text: str
    units: str | None = None


def split_sentences(paragraph: str) -> list[str]:
    return [sentence for sentence in SENTENCE_BOUNDARY_PATTERN.split(paragraph) if sentence]


def split_long_unit(unit: str, max_words: int) -> list[str]:
    """Hard-split a single sentence that alone exceeds `max_words`."""
    words = unit.split()
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def paragraph_units(paragraphs: list[str], max_words: int) -> list[tuple[int, str]]:
    """Break paragraphs into sentence units (tagged with their paragraph index).

    Sentence granularity is what makes overlap work: trailing sentences can be carried into
    the next chunk, whereas a whole paragraph rarely fits in the overlap budget.
    """
    units: list[tuple[int, str]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        for sentence in split_sentences(paragraph):
            if word_count(sentence) <= max_words:
                units.append((paragraph_index, sentence))
            else:
                units.extend(
                    (paragraph_index, piece) for piece in split_long_unit(sentence, max_words)
                )
    return units


def join_units(units: list[tuple[int, str]]) -> str:
    """Join units, keeping paragraph breaks between units from different paragraphs."""
    pieces: list[str] = []
    previous_paragraph: int | None = None
    for paragraph_index, unit in units:
        if pieces:
            pieces.append(" " if paragraph_index == previous_paragraph else "\n")
        pieces.append(unit)
        previous_paragraph = paragraph_index
    return "".join(pieces)


def pack_prose(paragraphs: list[str], config: ChunkingConfig) -> list[str]:
    """Greedily pack paragraph units into chunks with trailing-context overlap."""
    units = paragraph_units(paragraphs, config.max_words)
    chunks: list[str] = []
    current: list[tuple[int, str]] = []
    current_words = 0
    fresh_units = 0  # units added since the overlap was carried over
    for unit in units:
        unit_words = word_count(unit[1])
        if current and current_words + unit_words > config.max_words:
            chunks.append(join_units(current))
            overlap: list[tuple[int, str]] = []
            overlap_words = 0
            for previous in reversed(current):
                previous_words = word_count(previous[1])
                if overlap_words + previous_words > config.overlap_words:
                    break
                overlap.insert(0, previous)
                overlap_words += previous_words
            current, current_words, fresh_units = overlap, overlap_words, 0
        current.append(unit)
        current_words += unit_words
        fresh_units += 1
    if current and fresh_units:
        chunks.append(join_units(current))
    return chunks


def split_table_rows(rows: tuple[tuple[str, ...], ...], max_words: int) -> list[list[list[str]]]:
    """Split a table into row groups under `max_words`, repeating header rows in each group."""
    all_rows = [list(row) for row in rows]
    header_count = leading_header_rows(all_rows)
    header, body = all_rows[:header_count], all_rows[header_count:]
    header_words = sum(word_count(" ".join(row)) for row in header)
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    current_words = header_words
    for row in body:
        row_words = word_count(" ".join(row))
        if current and current_words + row_words > max_words:
            groups.append(header + current)
            current, current_words = [], header_words
        current.append(row)
        current_words += row_words
    if current or not groups:
        groups.append(header + current)
    return groups


def draft_section_chunks(section: Section, config: ChunkingConfig) -> list[DraftChunk]:
    """Prose chunks for the section's running text plus one or more chunks per table."""
    drafts: list[DraftChunk] = []
    paragraphs: list[str] = []
    recent_texts: list[str] = []
    for block in section.blocks:
        if block.kind == "text":
            paragraphs.append(block.text)
            recent_texts = [*recent_texts[-(UNITS_LOOKBACK_BLOCKS - 1) :], block.text]
            continue
        drafts.extend(table_drafts(section, block, recent_texts, config))
    drafts[:0] = [
        DraftChunk(section=section, chunk_type="prose", text=text)
        for text in pack_prose(paragraphs, config)
    ]
    return drafts


def table_drafts(
    section: Section, block: Block, recent_texts: list[str], config: ChunkingConfig
) -> list[DraftChunk]:
    first_cells = [cell for row in block.rows[:4] for cell in row]
    units = detect_units(first_cells) or detect_units(list(reversed(recent_texts)))
    title_candidates = [text for text in reversed(recent_texts) if len(text) <= MAX_TITLE_LENGTH]
    title = title_candidates[0] if title_candidates else ""
    drafts = []
    for group in split_table_rows(block.rows, config.table_max_words):
        body = render_rows(group)
        text = f"{title}\n{body}" if title else body
        drafts.append(DraftChunk(section=section, chunk_type="table", text=text, units=units))
    return drafts


def context_header(context: FilingContext, section: Section, units: str | None) -> str:
    record = context.record
    period = f"fiscal {context.fiscal_year} {context.fiscal_period}" if context.fiscal_year else ""
    parts = [
        f"{record.company_name} ({record.ticker}) {record.form} {period}".strip(),
        section.label,
    ]
    if units:
        parts.append(units)
    return " | ".join(parts)


def chunk_sections(
    sections: list[Section], context: FilingContext, config: ChunkingConfig
) -> tuple[list[Chunk], int]:
    """Chunk every section of a filing; returns (chunks, number of tiny drafts dropped)."""
    record = context.record
    chunks: list[Chunk] = []
    dropped_tiny = 0
    for section in sections:
        for draft in draft_section_chunks(section, config):
            if word_count(draft.text) < config.min_words:
                dropped_tiny += 1
                continue
            seq = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{record.accession}-{seq:05d}",
                    accession=record.accession,
                    ticker=record.ticker,
                    cik=record.cik,
                    company_name=record.company_name,
                    form=record.form,
                    is_amendment=record.is_amendment,
                    fiscal_year=context.fiscal_year,
                    fiscal_period=context.fiscal_period,
                    fiscal_year_end=record.fiscal_year_end,
                    period_of_report=record.report_date.isoformat() if record.report_date else None,
                    filing_date=record.filing_date.isoformat(),
                    part=section.part,
                    item=section.item,
                    section_title=section.title,
                    chunk_type=draft.chunk_type,
                    units=draft.units,
                    seq=seq,
                    text=draft.text,
                    search_text=f"{context_header(context, section, draft.units)}\n{draft.text}",
                    source_url=record.source_url,
                )
            )
    return chunks, dropped_tiny
