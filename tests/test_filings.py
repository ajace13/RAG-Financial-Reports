from datetime import date

from finrag.ingest.filings import (
    CompanyInfo,
    assign_fiscal_labels,
    parse_filing_table,
    select_filings,
)

COMPANY = CompanyInfo(ticker="MSFT", cik=789019, name="MICROSOFT CORP", fiscal_year_end="0630")

TABLE = {
    "form": ["8-K", "10-Q", "10-K", "10-K/A", "10-Q"],
    "accessionNumber": [
        "0000950170-24-000001",
        "0000950170-24-000002",
        "0000950170-24-000003",
        "0000950170-24-000004",
        "0000950170-21-000005",
    ],
    "filingDate": ["2024-10-30", "2024-10-30", "2024-07-30", "2024-08-15", "2021-04-27"],
    "reportDate": ["2024-10-30", "2024-09-30", "2024-06-30", "2024-06-30", "2021-03-31"],
    "primaryDocument": ["a.htm", "q.htm", "k.htm", "ka.htm", "old.htm"],
}


def labelled_records() -> list:
    return assign_fiscal_labels(parse_filing_table(TABLE, COMPANY), COMPANY.fiscal_year_end)


def test_parse_keeps_periodic_forms_and_labels_from_10k_history() -> None:
    records = labelled_records()
    assert [record.form for record in records] == ["10-Q", "10-K", "10-K/A", "10-Q"]
    assert (records[0].fiscal_year, records[0].fiscal_period) == (2025, "Q1")
    assert (records[1].fiscal_year, records[1].fiscal_period) == (2024, "FY")
    assert records[2].is_amendment
    # No 10-K precedes the 2021 quarter in this history, so the nominal MMDD fallback is used.
    assert (records[3].fiscal_year, records[3].fiscal_period) == (2021, "Q3")
    assert records[0].source_url == (
        "https://www.sec.gov/Archives/edgar/data/789019/000095017024000002/q.htm"
    )


def test_10k_history_beats_stale_fiscal_year_end() -> None:
    """A company that moved its year-end from June to December (EDGAR now reports 1231)."""
    company = CompanyInfo(ticker="AMCR", cik=1, name="Amcor", fiscal_year_end="1231")
    table = {
        "form": ["10-K", "10-Q"],
        "accessionNumber": ["A-1", "A-2"],
        "filingDate": ["2022-08-18", "2023-02-07"],
        "reportDate": ["2022-06-30", "2022-12-31"],
        "primaryDocument": ["k.htm", "q.htm"],
    }
    records = assign_fiscal_labels(parse_filing_table(table, company), company.fiscal_year_end)
    assert (records[1].fiscal_year, records[1].fiscal_period) == (2023, "Q2")


def test_select_filters_by_form_and_date() -> None:
    selected = select_filings(
        labelled_records(), forms={"10-K", "10-Q"}, filed_since=date(2023, 1, 1)
    )
    assert [record.accession for record in selected] == [
        "0000950170-24-000003",
        "0000950170-24-000002",
    ]
