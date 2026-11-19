from datetime import date

from finrag.chunking import (
    ChunkingConfig,
    FilingContext,
    chunk_sections,
    pack_prose,
    split_table_rows,
)
from finrag.ingest.models import FilingRecord
from finrag.parsing.html_blocks import Block
from finrag.parsing.sections import Section

RECORD = FilingRecord(
    ticker="MSFT",
    cik=789019,
    company_name="MICROSOFT CORP",
    form="10-K",
    accession="0000950170-24-087843",
    filing_date=date(2024, 7, 30),
    report_date=date(2024, 6, 30),
    primary_document="msft-20240630.htm",
    fiscal_year_end="0630",
    fiscal_year=2024,
    fiscal_period="FY",
)


def sentence(index: int, words: int = 10) -> str:
    return " ".join([f"Word{index}"] * words) + "."


def test_pack_prose_respects_limit_and_overlaps() -> None:
    paragraphs = [
        " ".join(sentence(index) for index in range(start, start + 3)) for start in (0, 3, 6)
    ]
    config = ChunkingConfig(max_words=40, overlap_words=10)
    chunks = pack_prose(paragraphs, config)
    assert all(len(chunk.split()) <= 40 for chunk in chunks)
    assert len(chunks) >= 2
    assert chunks[1].split()[0] == chunks[0].split()[-10]  # last unit carried over


def test_pack_prose_hard_splits_giant_sentence() -> None:
    chunks = pack_prose([" ".join(["x"] * 95)], ChunkingConfig(max_words=40, overlap_words=0))
    assert [len(chunk.split()) for chunk in chunks] == [40, 40, 15]


def test_split_table_rows_repeats_header() -> None:
    rows = (("", "2024", "2023"),) + tuple((f"Line {index}", "1", "2") for index in range(10))
    groups = split_table_rows(rows, max_words=15)
    assert len(groups) > 1
    assert all(group[0] == ["", "2024", "2023"] for group in groups)
    assert sum(len(group) - 1 for group in groups) == 10


def test_chunk_sections_builds_metadata_and_context_header() -> None:
    section = Section(
        part="II",
        item="7",
        title="Management's Discussion and Analysis",
        blocks=[
            Block(kind="text", text="(In millions)"),
            Block(kind="text", text="Revenue increased 16% driven by growth across each segment."),
            Block(
                kind="table",
                text="",
                rows=(("", "2024", "2023"), ("Revenue", "$245,122", "$211,915")),
            ),
            Block(kind="text", text="Tiny."),
        ],
    )
    context = FilingContext(record=RECORD, fiscal_year=2024, fiscal_period="FY")
    chunks, dropped = chunk_sections([section], context, ChunkingConfig(min_words=3))
    assert dropped == 0
    prose, table = chunks
    assert prose.chunk_type == "prose" and table.chunk_type == "table"
    assert table.units == "in millions"
    assert table.text.endswith("Revenue | $245,122 | $211,915")
    assert table.search_text.startswith(
        "MICROSOFT CORP (MSFT) 10-K fiscal 2024 FY | Part II, Item 7. "
        "Management's Discussion and Analysis | in millions"
    )
    assert [chunk.chunk_id for chunk in chunks] == [
        "0000950170-24-087843-00000",
        "0000950170-24-087843-00001",
    ]
