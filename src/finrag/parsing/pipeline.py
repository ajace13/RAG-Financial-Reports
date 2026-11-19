"""Filing -> chunks: parse HTML, finalize fiscal labels, split sections, chunk."""

from dataclasses import asdict, dataclass, field

from finrag.chunking import Chunk, ChunkingConfig, FilingContext, chunk_sections
from finrag.ingest.models import FilingRecord
from finrag.parsing.dei import fiscal_labels_from_dei
from finrag.parsing.html_blocks import parse_filing_html
from finrag.parsing.sections import UNSECTIONED, split_sections


@dataclass
class FilingParseReport:
    """Per-filing diagnostics written next to the chunks for inspection."""

    accession: str
    ticker: str
    form: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    label_source: str = "none"
    derived_fiscal_year: int | None = None
    derived_fiscal_period: str | None = None
    blocks: int = 0
    tables: int = 0
    sections: list[str] = field(default_factory=list)
    unsectioned_chunks: int = 0
    chunks: int = 0
    dropped_tiny_chunks: int = 0
    dropped_artifact_lines: int = 0
    dropped_running_header_lines: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_fiscal_labels(
    record: FilingRecord, dei: dict[str, str]
) -> tuple[int | None, str | None, str]:
    """Prefer the filing's own dei facts; fall back to labels derived at ingestion."""
    dei_year, dei_period = fiscal_labels_from_dei(dei)
    if dei_year is not None:
        expected_period = "FY" if record.form.startswith("10-K") else None
        return dei_year, dei_period or expected_period or record.fiscal_period, "dei"
    if record.fiscal_year is not None:
        return record.fiscal_year, record.fiscal_period, "derived"
    return None, None, "none"


def chunk_filing(
    record: FilingRecord, html: bytes, config: ChunkingConfig
) -> tuple[list[Chunk], FilingParseReport]:
    """Run the full parse for one filing and return its chunks plus a diagnostics report."""
    report = FilingParseReport(
        accession=record.accession,
        ticker=record.ticker,
        form=record.form,
        derived_fiscal_year=record.fiscal_year,
        derived_fiscal_period=record.fiscal_period,
    )
    document = parse_filing_html(html)
    fiscal_year, fiscal_period, label_source = resolve_fiscal_labels(record, document.dei)
    sections = split_sections(document.blocks, record.form)
    context = FilingContext(record=record, fiscal_year=fiscal_year, fiscal_period=fiscal_period)
    chunks, dropped_tiny = chunk_sections(sections, context, config)

    report.fiscal_year = fiscal_year
    report.fiscal_period = fiscal_period
    report.label_source = label_source
    report.blocks = len(document.blocks)
    report.tables = sum(1 for block in document.blocks if block.kind == "table")
    report.sections = [section.label for section in sections]
    report.unsectioned_chunks = sum(1 for chunk in chunks if chunk.item == UNSECTIONED)
    report.chunks = len(chunks)
    report.dropped_tiny_chunks = dropped_tiny
    report.dropped_artifact_lines = document.dropped_artifact_lines
    report.dropped_running_header_lines = document.dropped_running_header_lines
    return chunks, report
