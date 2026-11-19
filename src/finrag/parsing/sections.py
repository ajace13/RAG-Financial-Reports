"""Split a filing's blocks into Item sections (Item 1A Risk Factors, Item 7 MD&A, ...).

Heading detection is heuristic because formatting varies by company:

1. Candidates are short text lines that start with "Item <n>" followed by nothing, or by a
   title starting with a capital letter. This rejects prose such as "Item 7 of this report"
   and running headers such as "Item 1B, 1C".
2. Tables of contents are skipped: a run of 6+ candidates with almost no text between them.
   (Real headings can come 4-5 in a row with only "None." between them in a 10-Q's Part II.)
3. For each Item, the first remaining occurrence wins. Running page headers that repeat an
   Item number later on are mostly removed upstream, and the rest can only show up after the
   real heading.

10-Q item numbers restart in Part II (Part I Item 1 is Financial Statements, Part II Item 1
is Legal Proceedings), so 10-Q sections are keyed by (part, item).
"""

import logging
import re
from dataclasses import dataclass, field

from finrag.parsing.html_blocks import Block
from finrag.parsing.text_utils import word_count

logger = logging.getLogger(__name__)

ITEM_PATTERN = re.compile(
    r"^item\s*(?P<number>\d{1,2})(?P<letter>[a-c])?(?:[\s.:\-–—]+|$)(?P<rest>.*)$",
    re.IGNORECASE,
)
PART_PATTERN = re.compile(
    r"^part\s+(?P<roman>iv|iii|ii|i)\b\s*[.:\-–—]?\s*(?P<rest>.*)$", re.IGNORECASE
)
MAX_HEADING_LENGTH = 160
TOC_RUN_LENGTH = 6
TOC_MAX_WORDS_BETWEEN = 8

TEN_K_PARTS = {
    **dict.fromkeys(["1", "1A", "1B", "1C", "2", "3", "4"], "I"),
    **dict.fromkeys(["5", "6", "7", "7A", "8", "9", "9A", "9B", "9C"], "II"),
    **dict.fromkeys(["10", "11", "12", "13", "14"], "III"),
    **dict.fromkeys(["15", "16"], "IV"),
}
TEN_K_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
    "Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related "
    "Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TEN_Q_TITLES = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis of Financial Condition and Results of "
    "Operations",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings",
    ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities",
    ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information",
    ("II", "6"): "Exhibits",
}
FRONT_MATTER = "front"
UNSECTIONED = "unknown"
UNIDENTIFIED_TITLE = "Section not identified"
# Some filers (e.g. JPMorgan's 10-Qs) put MD&A and financial statements before the first
# Item heading. Front matter this large is mislabelled, so it is marked unidentified.
# (Large Item 15 sections are left alone: many filers legitimately place their financial
# statements under "Exhibits and Financial Statement Schedules".)
MAX_FRONT_MATTER_BLOCKS = 150


@dataclass
class Section:
    part: str
    item: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.item in (FRONT_MATTER, UNSECTIONED):
            return self.title
        prefix = f"Part {self.part}, " if self.part else ""
        return f"{prefix}Item {self.item}. {self.title}"


@dataclass(frozen=True)
class HeadingCandidate:
    block_index: int
    part: str
    item: str
    heading_text: str


def is_heading_rest(rest: str) -> bool:
    """A heading's remainder is empty or a title starting with a capital, quote or bracket."""
    return not rest or rest[0].isupper() or rest[0] in "[\"'“("


def parse_part(text: str) -> str | None:
    """Return the roman numeral of a 'PART II ...' heading line, else None."""
    if len(text) > MAX_HEADING_LENGTH:
        return None
    match = PART_PATTERN.match(text)
    if match is None or not is_heading_rest(match["rest"]):
        return None
    return match["roman"].upper()


def parse_item(text: str) -> tuple[str, str] | None:
    """Return (item number, rest-of-line) for an 'Item 7. Title' heading line, else None."""
    if len(text) > MAX_HEADING_LENGTH:
        return None
    match = ITEM_PATTERN.match(text)
    if match is None or not is_heading_rest(match["rest"]):
        return None
    item = match["number"] + (match["letter"] or "").upper()
    return item, match["rest"]


def item_sort_key(item: str) -> tuple[int, str]:
    digits = "".join(character for character in item if character.isdigit())
    return int(digits), item[len(digits) :]


def find_heading_candidates(blocks: list[Block], form: str) -> list[HeadingCandidate]:
    """Find Item heading lines, assigning parts (10-K from item number, 10-Q from context)."""
    is_quarterly = form.removesuffix("/A") == "10-Q"
    candidates: list[HeadingCandidate] = []
    current_part = "I"
    saw_part_since_last_item = False
    previous_item: str | None = None
    for index, block in enumerate(blocks):
        if block.kind != "text":
            continue
        part = parse_part(block.text)
        if part is not None:
            current_part = part
            saw_part_since_last_item = True
            continue
        parsed = parse_item(block.text)
        if parsed is None:
            continue
        item, _ = parsed
        if is_quarterly:
            # Without an explicit PART line, an item number going down means a new part
            # (Part I Item 4 -> Part II Item 1) or the end of the table of contents.
            if (
                previous_item is not None
                and not saw_part_since_last_item
                and item_sort_key(item) < item_sort_key(previous_item)
            ):
                current_part = "II" if current_part == "I" else "I"
            part_for_item = current_part
        else:
            part_for_item = TEN_K_PARTS.get(item, "")
        candidates.append(HeadingCandidate(index, part_for_item, item, block.text))
        previous_item = item
        saw_part_since_last_item = False
    return candidates


def words_between(blocks: list[Block], start: int, end: int) -> int:
    return sum(word_count(block.text) for block in blocks[start + 1 : end])


def drop_table_of_contents(
    candidates: list[HeadingCandidate], blocks: list[Block]
) -> list[HeadingCandidate]:
    """Remove runs of TOC_RUN_LENGTH+ candidates separated by almost no text.

    A run also ends when an Item repeats: a table of contents lists each Item once, so the
    real "Item 1" heading right after the TOC starts the body, not more TOC.
    """
    if not candidates:
        return candidates
    runs: list[list[int]] = []
    current_run = [0]
    keys_in_run = {(candidates[0].part, candidates[0].item)}
    for index in range(1, len(candidates)):
        previous, candidate = candidates[index - 1], candidates[index]
        key = (candidate.part, candidate.item)
        gap = words_between(blocks, previous.block_index, candidate.block_index)
        if gap <= TOC_MAX_WORDS_BETWEEN and key not in keys_in_run:
            current_run.append(index)
            keys_in_run.add(key)
        else:
            runs.append(current_run)
            current_run = [index]
            keys_in_run = {key}
    runs.append(current_run)
    toc_indices = {index for run in runs if len(run) >= TOC_RUN_LENGTH for index in run}
    return [candidate for index, candidate in enumerate(candidates) if index not in toc_indices]


def section_title(form: str, part: str, item: str, heading_text: str) -> str:
    """Standard SEC title for the item; falls back to the heading's own text.

    10-K Item 6 changed meaning in 2021 (Selected Financial Data -> [Reserved]), so its
    title is taken from the filing itself.
    """
    parsed = parse_item(heading_text)
    own_title = parsed[1].rstrip(".") if parsed and parsed[1] else ""
    if form.removesuffix("/A") == "10-Q":
        standard = TEN_Q_TITLES.get((part, item))
    elif item == "6" and own_title:
        standard = own_title.title() if own_title.isupper() else own_title
    else:
        standard = TEN_K_TITLES.get(item)
    return standard or own_title or f"Item {item}"


def split_sections(blocks: list[Block], form: str) -> list[Section]:
    """Group blocks into sections in document order. Text before the first heading is front
    matter; a document with no detectable headings becomes a single unsectioned section."""
    candidates = drop_table_of_contents(find_heading_candidates(blocks, form), blocks)
    first_by_key: dict[tuple[str, str], HeadingCandidate] = {}
    for candidate in candidates:
        first_by_key.setdefault((candidate.part, candidate.item), candidate)
    headings = sorted(first_by_key.values(), key=lambda candidate: candidate.block_index)

    if not headings:
        logger.debug("No Item headings detected; treating document as one section")
        return [Section(part="", item=UNSECTIONED, title="Full document", blocks=list(blocks))]

    sections = []
    if headings[0].block_index > 0:
        sections.append(
            Section(
                part="",
                item=FRONT_MATTER,
                title="Cover page and front matter",
                blocks=blocks[: headings[0].block_index],
            )
        )
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        end = following.block_index if following is not None else len(blocks)
        sections.append(
            Section(
                part=heading.part,
                item=heading.item,
                title=section_title(form, heading.part, heading.item, heading.heading_text),
                blocks=blocks[heading.block_index + 1 : end],
            )
        )
    return [relabel_if_oversized(section) for section in sections]


def relabel_if_oversized(section: Section) -> Section:
    """Mark front matter that is far too large to be a cover page as unidentified."""
    if section.item != FRONT_MATTER or len(section.blocks) <= MAX_FRONT_MATTER_BLOCKS:
        return section
    logger.debug("Relabelling oversized section %s (%d blocks)", section.label, len(section.blocks))
    return Section(part="", item=UNSECTIONED, title=UNIDENTIFIED_TITLE, blocks=section.blocks)
