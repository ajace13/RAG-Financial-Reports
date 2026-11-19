"""Read document-level inline-XBRL `dei` facts (fiscal year focus, period, form type).

A regex is enough here: we need a handful of tagged values, and it is ~100x faster than
building a full parse tree when we only want to verify which fiscal period a filing covers.
"""

import html
import re

# Only opening tags are matched: facts can be nested inside other facts, so a single
# open..close pattern would swallow the inner one.
DEI_OPEN_TAG_PATTERN = re.compile(
    r"<ix:nonnumeric\b[^>]*\bname=\"(dei:[A-Za-z]+)\"[^>]*>", re.IGNORECASE
)
CLOSE_TAG_PATTERN = re.compile(r"</ix:nonnumeric>", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

FISCAL_YEAR_FOCUS = "dei:DocumentFiscalYearFocus"
FISCAL_PERIOD_FOCUS = "dei:DocumentFiscalPeriodFocus"


def read_dei_facts(document: bytes | str) -> dict[str, str]:
    """Return the first value of each `dei:*` fact in an inline-XBRL document."""
    text = document.decode("utf-8", errors="replace") if isinstance(document, bytes) else document
    facts: dict[str, str] = {}
    for opening in DEI_OPEN_TAG_PATTERN.finditer(text):
        name = opening.group(1)
        if name in facts:
            continue
        closing = CLOSE_TAG_PATTERN.search(text, opening.end())
        if closing is None:
            continue
        inner = text[opening.end() : closing.start()]
        value = html.unescape(TAG_PATTERN.sub(" ", inner))
        facts[name] = WHITESPACE_PATTERN.sub(" ", value).strip()
    return facts


def fiscal_labels_from_dei(facts: dict[str, str]) -> tuple[int | None, str | None]:
    """Fiscal (year, period) from dei facts, or (None, None) when absent or malformed."""
    year_text = facts.get(FISCAL_YEAR_FOCUS, "")
    period = facts.get(FISCAL_PERIOD_FOCUS, "").upper() or None
    year = int(year_text) if year_text.isdigit() and len(year_text) == 4 else None
    if period not in {"FY", "Q1", "Q2", "Q3", "Q4"}:
        period = None
    return year, period
