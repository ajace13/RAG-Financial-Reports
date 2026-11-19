from finrag.parsing.html_blocks import Block
from finrag.parsing.sections import FRONT_MATTER, UNSECTIONED, parse_item, split_sections


def text_blocks(*lines: str) -> list[Block]:
    return [Block(kind="text", text=line) for line in lines]


def test_parse_item_accepts_headings_and_rejects_prose() -> None:
    assert parse_item("ITEM 7. MANAGEMENT'S DISCUSSION") == ("7", "MANAGEMENT'S DISCUSSION")
    assert parse_item("Item 1A") == ("1A", "")
    assert parse_item("Item 6. [Reserved]") == ("6", "[Reserved]")
    assert parse_item("Item 7 of this report discusses results") is None
    assert parse_item("Item 1B, 1C") is None
    assert parse_item("Item 1. - Business") == ("1", "Business")


def test_split_10k_skips_table_of_contents_and_keeps_first_real_heading() -> None:
    toc = [f"Item {number}. Title {number}" for number in ["1", "1A", "2", "3", "5", "7", "8"]]
    blocks = text_blocks(
        "Annual report cover",
        *toc,
        "ITEM 1. BUSINESS",
        "We make software and devices for consumers and businesses around the world.",
        "ITEM 7. MANAGEMENT'S DISCUSSION",
        "Revenue increased sixteen percent, driven by growth across each segment of the business.",
        "Item 7",  # stray running header left after the real heading
        "More analysis of operating income and expenses for the fiscal year presented here.",
    )
    sections = split_sections(blocks, "10-K")
    assert [(section.item, section.part) for section in sections] == [
        (FRONT_MATTER, ""),
        ("1", "I"),
        ("7", "II"),
    ]
    assert sections[2].title.startswith("Management's Discussion")
    assert len(sections[2].blocks) == 3


def test_split_10q_keys_items_by_part() -> None:
    blocks = text_blocks(
        "PART I. FINANCIAL INFORMATION",
        "Item 1. Financial Statements",
        "Balance sheet narrative with enough words to count as real content here.",
        "Item 2. Management's Discussion",
        "Quarterly results narrative with enough words to count as real content here.",
        "Item 1. Legal Proceedings",  # no PART II line: the number going down implies Part II
        "See Note 12 for a description of legal proceedings and related contingencies.",
    )
    sections = split_sections(blocks, "10-Q")
    assert [(section.part, section.item, section.title) for section in sections] == [
        ("", FRONT_MATTER, "Cover page and front matter"),
        ("I", "1", "Financial Statements"),
        (
            "I",
            "2",
            "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        ),
        ("II", "1", "Legal Proceedings"),
    ]


def test_document_without_headings_is_one_unsectioned_section() -> None:
    sections = split_sections(text_blocks("Just some text", "and more text"), "10-K")
    assert len(sections) == 1 and sections[0].item == UNSECTIONED
