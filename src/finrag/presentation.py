"""Rendering helpers for the Streamlit app, kept framework-free so they can be unit-tested.

Two rendering traps this module handles:
- Streamlit markdown treats `$...$` as LaTeX, so "$1,577 | $1,373" would render as math.
  Answer text gets its dollar signs escaped, and passages are rendered as escaped HTML.
- EDGAR links use URL text fragments (`#:~:text=start,end`), which Chromium-based browsers
  scroll to and highlight. Fragments must match the rendered page text, so table rows link
  by their row label only (the page has no "|" separators).
"""

import html
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

from finrag.chunking import split_sentences
from finrag.generate.models import Answer, Passage
from finrag.generate.structured import normalize_for_match
from finrag.retrieve.models import SearchHit

YEAR_MENTION_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
CELL_SEPARATOR = " | "
NUMERIC_CELL_PATTERN = re.compile(r"^[$€£(\-–—]*\s*\d[\d,.]*\s*\)?\s*%?$")
FRAGMENT_EDGE_WORDS = 5

PASSAGE_STYLE = """
<style>
.finrag-passage { font-size: 0.92rem; line-height: 1.5; }
.finrag-passage p { margin: 0 0 0.6em 0; }
.finrag-passage mark { background: rgba(255, 213, 79, 0.45); color: inherit; padding: 0 2px; }
.finrag-passage table { border-collapse: collapse; width: 100%; }
.finrag-passage table { font-variant-numeric: tabular-nums; }
.finrag-passage td { border-bottom: 1px solid rgba(128, 128, 128, 0.25); padding: 3px 8px; }
.finrag-passage td.num { text-align: right; white-space: nowrap; }
.finrag-passage tr.cited td { background: rgba(255, 213, 79, 0.35); }
.finrag-passage .caption { font-weight: 600; margin-bottom: 0.4em; }
.finrag-passage .table-scroll { overflow-x: auto; max-width: 100%; }
</style>
"""


def escape_markdown(text: str) -> str:
    """Escape characters Streamlit markdown would interpret: '$' (LaTeX) and '*'/'_' runs."""
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")


def citation_numbering(answer: Answer) -> dict[str, int]:
    """Passage id -> citation number, in order of first citation."""
    return {passage_id: number for number, passage_id in enumerate(answer.cited_passage_ids, 1)}


def answer_markdown(answer: Answer) -> str:
    """Answer text with [n] markers after each cited segment; '?' marks unverified quotes."""
    numbering = citation_numbering(answer)
    pieces = []
    for segment in answer.segments:
        markers = "".join(
            f"[{numbering[citation.passage_id]}{'' if citation.verified else '?'}]"
            for citation in segment.citations
        )
        pieces.append(escape_markdown(segment.text) + (f" **{markers}**" if markers else ""))
    return "".join(pieces).strip()


def cited_texts_for(answer: Answer, passage_id: str) -> list[str]:
    return [c.cited_text for c in answer.citations if c.passage_id == passage_id and c.cited_text]


def is_cited(unit: str, cited_texts: Sequence[str]) -> bool:
    """A block/sentence is cited if it lies inside a cited span or a quote lies inside it."""
    normalized_unit = normalize_for_match(unit)
    if not normalized_unit:
        return False
    for cited in cited_texts:
        normalized_cited = normalize_for_match(cited)
        if normalized_cited and (
            normalized_unit in normalized_cited or normalized_cited in normalized_unit
        ):
            return True
    return False


def render_table_html(lines: Sequence[str], cited_texts: Sequence[str]) -> str:
    rows, captions = [], []
    for line in lines:
        if CELL_SEPARATOR not in line:
            captions.append(f'<div class="caption">{html.escape(line)}</div>')
            continue
        cells = "".join(
            f'<td class="num">{html.escape(cell)}</td>'
            if NUMERIC_CELL_PATTERN.match(cell)
            else f"<td>{html.escape(cell)}</td>"
            for cell in line.split(CELL_SEPARATOR)
        )
        row_class = ' class="cited"' if is_cited(line, cited_texts) else ""
        rows.append(f"<tr{row_class}>{cells}</tr>")
    table = f'<div class="table-scroll"><table>{"".join(rows)}</table></div>' if rows else ""
    return "".join(captions) + table


def render_prose_html(text: str, cited_texts: Sequence[str]) -> str:
    paragraphs = []
    for line in (line.strip() for line in text.split("\n")):
        if not line:
            continue
        sentences = [
            f"<mark>{html.escape(sentence)}</mark>"
            if is_cited(sentence, cited_texts)
            else html.escape(sentence)
            for sentence in split_sentences(line)
        ]
        paragraphs.append(f"<p>{' '.join(sentences)}</p>")
    return "".join(paragraphs)


def render_passage_html(passage: Passage, cited_texts: Sequence[str], chunk_type: str) -> str:
    """Passage body as HTML with cited rows/sentences highlighted."""
    lines = [line.strip() for line in passage.text.split("\n") if line.strip()]
    body = (
        render_table_html(lines, cited_texts)
        if chunk_type == "table"
        else render_prose_html(passage.text, cited_texts)
    )
    return f'{PASSAGE_STYLE}<div class="finrag-passage">{body}</div>'


def fragment_anchor_text(unit: str) -> str:
    """Text a browser can find on the EDGAR page: a table row's label, or the sentence."""
    return unit.split(CELL_SEPARATOR)[0].strip() if CELL_SEPARATOR in unit else unit.strip()


def encode_fragment_term(text: str) -> str:
    """Percent-encode a text-fragment term. '-', ',' and '&' are fragment syntax and must be
    encoded; urllib's quote() leaves '-' alone because it is URL-safe elsewhere."""
    return quote(text, safe="").replace("-", "%2D")


def text_fragment_url(source_url: str, anchor_text: str) -> str:
    """EDGAR URL that scrolls to and highlights `anchor_text` (text fragments, Chromium)."""
    words = anchor_text.split()
    if not words:
        return source_url
    if len(words) <= 2 * FRAGMENT_EDGE_WORDS:
        directive = encode_fragment_term(" ".join(words))
    else:
        start = encode_fragment_term(" ".join(words[:FRAGMENT_EDGE_WORDS]))
        end = encode_fragment_term(" ".join(words[-FRAGMENT_EDGE_WORDS:]))
        directive = f"{start},{end}"
    return f"{source_url}#:~:text={directive}"


def passage_link(passage: Passage, cited_texts: Sequence[str]) -> str:
    """Deep link to the first cited block of the passage (or the passage's first block)."""
    cited_blocks = [block for block in passage.blocks if is_cited(block, cited_texts)]
    candidates = cited_blocks or list(passage.blocks)
    anchor = next((fragment_anchor_text(block) for block in candidates if block.strip()), "")
    return text_fragment_url(passage.source_url, anchor)


def fiscal_years_mentioned(question: str) -> set[int]:
    """Four-digit years in a question ('FY2018', 'fiscal 2012', 'in 2009')."""
    return {int(year) for year in YEAR_MENTION_PATTERN.findall(question)}


def years_outside_coverage(
    mentioned: set[int], indexed_years: Sequence[int]
) -> tuple[set[int], set[int]]:
    """Mentioned years that are (older than, newer than) every indexed fiscal year."""
    if not indexed_years:
        return set(), set()
    oldest, newest = min(indexed_years), max(indexed_years)
    return {year for year in mentioned if year < oldest}, {
        year for year in mentioned if year > newest
    }


def debug_rows(hits: Sequence[SearchHit]) -> list[dict[str, Any]]:
    """One row per retrieved passage with its rank in each retrieval stage."""
    rows = []
    for position, hit in enumerate(hits, start=1):
        document = hit.document
        rows.append(
            {
                "final rank": position,
                "filing": f"{document['ticker']} {document['form']} FY{document['fiscal_year']}",
                "section": document.get("section_title", ""),
                "type": document.get("chunk_type", ""),
                "BM25 rank": hit.ranks.get("bm25"),
                "vector rank": hit.ranks.get("vector"),
                "RRF rank": hit.ranks.get("rrf"),
                "rerank score": round(hit.scores["rerank"], 3) if "rerank" in hit.scores else None,
                "chunk id": hit.chunk_id,
            }
        )
    return rows
