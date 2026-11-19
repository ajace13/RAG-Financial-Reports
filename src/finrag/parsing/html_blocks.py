"""Parse filing HTML into an ordered list of text and table blocks.

Approach: drop hidden content (the inline-XBRL header, display:none elements, scripts),
pull each top-level table out into its own block, then put line breaks at block-level
element boundaries so `get_text()` gives one line per paragraph. Without that, text from
adjacent <span>s inside one <p> would be split apart, or separate <div>s would run
together.
"""

import logging
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from finrag.parsing.dei import read_dei_facts
from finrag.parsing.tables import extract_table_rows, is_layout_table, render_rows
from finrag.parsing.text_utils import normalize_whitespace

logger = logging.getLogger(__name__)

# Inline-XBRL filings start with an XML declaration; the HTML parser handles them correctly.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = [
    "p", "div", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "center", "blockquote", "pre", "section", "article", "dt", "dd",
]  # fmt: skip
REMOVED_TAGS = ["ix:header", "script", "style", "head", "title", "noscript"]
HIDDEN_STYLE_PATTERN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
TABLE_PLACEHOLDER = "⁣TABLE{index}⁣"
TABLE_PLACEHOLDER_PATTERN = re.compile("⁣TABLE(\\d+)⁣")

PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s+)?(?:\d{1,3}|[ivxlc]{1,6})$", re.IGNORECASE)
TABLE_OF_CONTENTS_PATTERN = re.compile(r"^(?:table\s+of\s+contents|index)$", re.IGNORECASE)
# Short lines repeated this often are running page headers/footers (e.g. "PART II" / "Item 8"
# printed atop every page), not content.
RUNNING_HEADER_MIN_REPEATS = 5
RUNNING_HEADER_MAX_LENGTH = 60


@dataclass(frozen=True)
class Block:
    """A paragraph of text, or a table rendered as pipe-delimited rows."""

    kind: Literal["text", "table"]
    text: str
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass
class ParsedDocument:
    blocks: list[Block]
    dei: dict[str, str] = field(default_factory=dict)
    dropped_artifact_lines: int = 0
    dropped_running_header_lines: int = 0


def remove_hidden_content(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(REMOVED_TAGS):
        tag.decompose()
    for tag in soup.find_all(style=HIDDEN_STYLE_PATTERN):
        tag.decompose()


def replace_tables_with_placeholders(soup: BeautifulSoup) -> list[Block]:
    """Swap each outermost <table> for a placeholder line; return the blocks they become.

    Layout tables (one cell per row) become text blocks so bullets and banners read as prose.
    """
    table_blocks: list[Block] = []
    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue
        rows = extract_table_rows(table)
        if not rows:
            table.decompose()
            continue
        if is_layout_table(rows):
            block = Block(kind="text", text="\n".join(row[0] for row in rows))
        elif len(rows) == 1:
            # One-row "tables" are layout too, e.g. headings laid out as "ITEM 7. | MD&A".
            block = Block(kind="text", text=" ".join(rows[0]))
        else:
            block = Block(
                kind="table", text=render_rows(rows), rows=tuple(tuple(row) for row in rows)
            )
        placeholder = TABLE_PLACEHOLDER.format(index=len(table_blocks))
        table.replace_with(NavigableString(f"\n{placeholder}\n"))
        table_blocks.append(block)
    return table_blocks


def mark_block_boundaries(soup: BeautifulSoup) -> None:
    """Insert newlines around block-level elements so each paragraph becomes one line."""
    for line_break in soup.find_all("br"):
        line_break.replace_with(NavigableString("\n"))
    for tag in soup.find_all(BLOCK_TAGS):
        if isinstance(tag, Tag):
            tag.insert_before(NavigableString("\n"))
            tag.insert_after(NavigableString("\n"))


def is_page_artifact(line: str) -> bool:
    """Page numbers and 'Table of Contents' back-links repeated on every printed page."""
    return bool(PAGE_NUMBER_PATTERN.match(line) or TABLE_OF_CONTENTS_PATTERN.match(line))


def find_running_headers(lines: list[str]) -> set[str]:
    """Short lines that repeat many times across the document."""
    counts = Counter(line for line in lines if len(line) <= RUNNING_HEADER_MAX_LENGTH)
    return {line for line, count in counts.items() if count >= RUNNING_HEADER_MIN_REPEATS}


def parse_filing_html(html: bytes | str) -> ParsedDocument:
    """Parse a filing's primary document into ordered blocks plus document-level dei facts."""
    dei = read_dei_facts(html)
    soup = BeautifulSoup(html, "lxml")
    remove_hidden_content(soup)
    table_blocks = replace_tables_with_placeholders(soup)
    mark_block_boundaries(soup)

    root = soup.body or soup
    lines = [normalize_whitespace(raw_line) for raw_line in root.get_text().split("\n")]
    lines = [line for line in lines if line]
    running_headers = find_running_headers(lines)

    blocks: list[Block] = []
    dropped_artifacts = 0
    dropped_running_headers = 0
    for line in lines:
        placeholder = TABLE_PLACEHOLDER_PATTERN.fullmatch(line)
        if placeholder:
            blocks.append(table_blocks[int(placeholder.group(1))])
        elif is_page_artifact(line):
            dropped_artifacts += 1
        elif line in running_headers:
            dropped_running_headers += 1
        else:
            blocks.append(Block(kind="text", text=line))
    return ParsedDocument(
        blocks=blocks,
        dei=dei,
        dropped_artifact_lines=dropped_artifacts,
        dropped_running_header_lines=dropped_running_headers,
    )
