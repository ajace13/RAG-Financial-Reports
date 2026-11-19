"""Earnings-release 8-Ks and DEF 14A proxies: discovery, fiscal labels, exhibits, sections."""

from dataclasses import replace
from datetime import date

import pytest

from finrag.generate.passages import passage_label
from finrag.ingest.filings import (
    CompanyInfo,
    assign_fiscal_labels,
    find_exhibit_document,
    is_earnings_release,
    parse_filing_table,
)
from finrag.ingest.fiscal import earnings_release_period, latest_year_end_before
from finrag.ingest.service import (
    IngestProgress,
    IngestRequest,
    attach_release_exhibits,
    select_requested_filings,
)
from finrag.parsing.html_blocks import Block
from finrag.parsing.sections import document_sections

MSFT = CompanyInfo(ticker="MSFT", cik=789019, name="MICROSOFT CORP", fiscal_year_end="0630")
TABLE = {
    "form": ["10-K", "8-K", "8-K", "DEF 14A", "10-Q", "8-K"],
    "accessionNumber": ["K-25", "E-Q1", "O-1", "P-25", "Q-Q1", "E-Q4"],
    "filingDate": [
        "2025-07-30",
        "2025-10-29",
        "2025-11-04",
        "2025-10-21",
        "2025-10-29",
        "2025-07-30",
    ],
    "reportDate": ["2025-06-30", "2025-10-29", "2025-11-04", "", "2025-09-30", "2025-07-30"],
    "items": ["", "2.02,9.01", "8.01,9.01", "", "", "2.02,9.01"],
    "primaryDocument": ["k.htm", "e.htm", "o.htm", "p.htm", "q.htm", "e4.htm"],
}


def labelled() -> dict[str, object]:
    records = assign_fiscal_labels(parse_filing_table(TABLE, MSFT), MSFT.fiscal_year_end)
    return {record.accession: record for record in records}


def test_parse_keeps_8k_items_and_proxies() -> None:
    records = labelled()
    assert records["E-Q1"].items == "2.02,9.01"
    assert is_earnings_release(records["E-Q1"])
    assert not is_earnings_release(records["O-1"])
    assert records["P-25"].form == "DEF 14A"


def test_fiscal_labels_for_releases_and_proxies() -> None:
    records = labelled()
    # Q1 FY2026 results (quarter ended 2025-09-30), released 2025-10-29.
    assert (records["E-Q1"].fiscal_year, records["E-Q1"].fiscal_period) == (2026, "Q1")
    # Q4 FY2025 results released 2025-07-30, the same day the FY2025 10-K was filed.
    assert (records["E-Q4"].fiscal_year, records["E-Q4"].fiscal_period) == (2025, "Q4")
    # The October 2025 proxy covers fiscal 2025 (pay for the year that ended in June).
    assert (records["P-25"].fiscal_year, records["P-25"].fiscal_period) == (2025, "FY")
    # Non-earnings 8-Ks get no labels.
    assert records["O-1"].fiscal_year is None


@pytest.mark.parametrize(
    ("filing_date", "year_ends", "fiscal_year_end", "expected"),
    [
        (date(2026, 1, 28), [date(2024, 12, 31)], "1231", date(2025, 12, 31)),  # 10-K not filed yet
        (date(2023, 2, 7), [date(2022, 6, 30)], "1231", date(2022, 6, 30)),  # stale MMDD ignored
        (date(2025, 10, 29), [date(2025, 6, 30)], "0630", date(2025, 6, 30)),
    ],
)
def test_latest_year_end_before(
    filing_date: date, year_ends: list[date], fiscal_year_end: str, expected: date
) -> None:
    assert latest_year_end_before(filing_date, year_ends, fiscal_year_end) == expected


@pytest.mark.parametrize(
    ("filing_date", "prior_year_end", "expected"),
    [
        (date(2026, 1, 28), date(2025, 12, 31), (2025, "Q4")),
        (date(2025, 4, 24), date(2024, 12, 31), (2025, "Q1")),
        (date(2025, 10, 8), date(2024, 12, 28), (2025, "Q3")),  # PepsiCo-style 36-week Q3
        (date(2023, 2, 7), date(2022, 6, 30), (2023, "Q2")),  # Amcor fiscal 2023 Q2
    ],
)
def test_earnings_release_period(
    filing_date: date, prior_year_end: date, expected: tuple[int, str]
) -> None:
    assert earnings_release_period(filing_date, prior_year_end) == expected


INDEX_ROWS = [
    ("1", "8-K", '<a href="x">msft-20261028.htm</a> &nbsp;&nbsp;iXBRL', "8-K"),
    ("2", "EX-99.1", '<a href="y">msft-ex99_1.htm</a>', "EX-99.1"),
    ("3", "EX-99.2", '<a href="z">msft-ex99_2.htm</a>', "EX-99.2"),
]
INDEX_HTML = (
    '<table class="tableFile"><tr><th>Seq</th><th>Description</th><th>Document</th>'
    "<th>Type</th><th>Size</th></tr>"
    + "".join(
        f"<tr><td>{seq}</td><td>{description}</td><td>{document}</td><td>{kind}</td><td>1</td></tr>"
        for seq, description, document, kind in INDEX_ROWS
    )
    + "</table>"
)


def test_find_exhibit_document() -> None:
    assert find_exhibit_document(INDEX_HTML) == "msft-ex99_1.htm"
    only_ex99 = INDEX_HTML.replace("EX-99.1", "EX-99").replace("EX-99.2", "EX-98")
    assert find_exhibit_document(only_ex99) == "msft-ex99_1.htm"
    assert find_exhibit_document("<table><tr><td>1</td></tr></table>") is None


class FakeEdgar:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def get_bytes(self, url: str) -> bytes:
        return self.pages[url].encode()


def test_attach_release_exhibits_swaps_in_press_release() -> None:
    records = labelled()
    release, report_10q = records["E-Q1"], records["Q-Q1"]
    no_exhibit = replace(release, accession="E-NONE")
    client = FakeEdgar({release.index_url: INDEX_HTML, no_exhibit.index_url: "<table></table>"})
    progress: list[IngestProgress] = []
    resolved, missing = attach_release_exhibits(
        client, [report_10q, release, no_exhibit], progress.append
    )
    assert [record.primary_document for record in resolved] == ["q.htm", "msft-ex99_1.htm"]
    assert missing == 1
    assert resolved[1].source_url.endswith("/E-Q1/msft-ex99_1.htm".replace("E-Q1", "EQ1"))


def test_request_forms_scope_and_selection() -> None:
    plain = IngestRequest("MSFT")
    extended = IngestRequest("MSFT", include_earnings_releases=True, include_proxies=True)
    assert plain.forms == ("10-K", "10-Q")
    assert extended.forms == ("10-K", "10-Q", "8-K", "DEF 14A")
    assert extended.scope_label == "last 5 years + earnings releases + proxies"
    selected = select_requested_filings(list(labelled().values()), extended, date(2025, 1, 1))
    assert {record.accession for record in selected} == {"K-25", "E-Q1", "P-25", "Q-Q1", "E-Q4"}


def test_whole_document_sections_and_labels() -> None:
    blocks = [
        Block(kind="text", text="Item 1. Election of Directors"),
        Block(kind="text", text="x"),
    ]
    (release,) = document_sections(blocks, "8-K")
    (proxy,) = document_sections(blocks, "DEF 14A")
    assert (release.item, release.label) == ("release", "Earnings release (Exhibit 99.1)")
    assert (proxy.item, proxy.label) == ("proxy", "Proxy statement")
    assert len(proxy.blocks) == 2
    label = passage_label(
        {
            "company_name": "MICROSOFT CORP",
            "ticker": "MSFT",
            "form": "8-K",
            "fiscal_year": 2026,
            "fiscal_period": "Q1",
            "item": "release",
            "section_title": "Earnings release (Exhibit 99.1)",
        }
    )
    assert label == "MICROSOFT CORP (MSFT) 8-K FY2026 Q1 | Earnings release (Exhibit 99.1)"
