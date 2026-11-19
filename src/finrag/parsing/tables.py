"""Convert EDGAR HTML tables into compact pipe-delimited rows.

EDGAR financial tables are layout-heavy: spacer cells, `$` and `)` in their own cells,
and colspans used for alignment. We do not try to rebuild a perfect grid. For retrieval
what matters is that each row's label stays next to its values, so we merge symbol cells
into their neighbours and drop empty cells.
"""

import re

from bs4 import Tag

from finrag.parsing.text_utils import normalize_whitespace

PREFIX_SYMBOLS = frozenset({"$", "€", "£", "¥"})
SUFFIX_SYMBOLS = frozenset({")", "%", ")%", "%)", "pts", "bps"})
BULLET_SYMBOLS = frozenset({"•", "●", "◦", "▪", "■", "·", "–", "—", "-", "o"})

YEAR_PATTERN = re.compile(r"^(?:fiscal\s+)?(?:19|20)\d{2}$", re.IGNORECASE)
NUMERIC_VALUE_PATTERN = re.compile(r"^[$€£¥(\-–—]*\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*%?$")
UNITS_PATTERN = re.compile(r"\b(?:in\s+)?(millions|thousands|billions)\b", re.IGNORECASE)
MAX_UNITS_TEXT_LENGTH = 150
MAX_HEADER_ROWS = 4


OPEN_PAREN_SPACE_PATTERN = re.compile(r"\(\s+")


def merge_symbol_cells(cells: list[str]) -> list[str]:
    """Attach currency symbols to the following value and ')' / '%' to the preceding one.

    Empty cells are dropped. Bullet cells become a prefix of the text that follows them.
    Negative numbers written '( 76,421)' are tightened to '(76,421)'.
    """
    merged: list[str] = []
    pending_prefix = ""
    for raw_cell in cells:
        cell = OPEN_PAREN_SPACE_PATTERN.sub("(", raw_cell)
        if not cell:
            continue
        if cell in PREFIX_SYMBOLS:
            pending_prefix = cell
        elif cell in BULLET_SYMBOLS:
            pending_prefix = cell + " "
        elif cell in SUFFIX_SYMBOLS and merged:
            merged[-1] += cell
        else:
            merged.append(pending_prefix + cell)
            pending_prefix = ""
    return merged


def extract_table_rows(table: Tag) -> list[list[str]]:
    """Return cleaned cell texts for each non-empty row belonging directly to `table`."""
    rows = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue  # row of a nested table; its text is already inside an outer cell
        cells = [
            normalize_whitespace(cell.get_text(" "))
            for cell in row.find_all(["td", "th"], recursive=False)
        ]
        cleaned = merge_symbol_cells(cells)
        if cleaned:
            rows.append(cleaned)
    return rows


def is_layout_table(rows: list[list[str]]) -> bool:
    """A table with at most one cell per row is page layout (bullets, banners), not data."""
    return all(len(row) <= 1 for row in rows)


def render_rows(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def is_header_row(row: list[str]) -> bool:
    """Header rows hold labels or years but no numeric values ('Year Ended June 30,' is fine)."""
    return all(YEAR_PATTERN.match(cell) or not NUMERIC_VALUE_PATTERN.match(cell) for cell in row)


def leading_header_rows(rows: list[list[str]]) -> int:
    """Count consecutive header rows at the top of a table (capped at MAX_HEADER_ROWS)."""
    count = 0
    for row in rows[:MAX_HEADER_ROWS]:
        if not is_header_row(row):
            break
        count += 1
    return count if count < len(rows) else 0


def detect_units(texts: list[str]) -> str | None:
    """Find a units phrase such as 'in millions' in short label-like texts."""
    for text in texts:
        if len(text) > MAX_UNITS_TEXT_LENGTH:
            continue
        match = UNITS_PATTERN.search(text)
        if match:
            return f"in {match.group(1).lower()}"
    return None
