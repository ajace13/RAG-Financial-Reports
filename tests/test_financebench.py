from dataclasses import replace
from datetime import date

from finrag.evaluation.financebench import candidate_filings, dei_confirms, parse_doc_name
from finrag.ingest.models import FilingRecord

BASE_RECORD = FilingRecord(
    ticker="ULTA",
    cik=1403568,
    company_name="Ulta Beauty, Inc.",
    form="10-K",
    accession="U-2024",
    filing_date=date(2024, 3, 26),
    report_date=date(2024, 2, 3),
    primary_document="k.htm",
    fiscal_year_end="0130",
    fiscal_year=2024,  # derived with the end-year convention
    fiscal_period="FY",
)


def test_parse_annual_doc_name() -> None:
    doc = parse_doc_name("JOHNSON_JOHNSON_2022_10K")
    assert doc is not None
    assert (doc.company_key, doc.ticker, doc.year, doc.quarter) == (
        "JOHNSON_JOHNSON",
        "JNJ",
        2022,
        None,
    )
    assert (doc.form, doc.fiscal_period) == ("10-K", "FY")


def test_parse_quarterly_and_non_periodic_doc_names() -> None:
    quarterly = parse_doc_name("Pfizer_2023Q2_10Q")
    assert quarterly is not None
    assert (quarterly.ticker, quarterly.form, quarterly.fiscal_period) == ("PFE", "10-Q", "Q2")
    earnings = parse_doc_name("AMCOR_2023Q4_EARNINGS")
    assert earnings is not None and earnings.form is None
    eight_k = parse_doc_name("AMCOR_2022_8K_dated-2022-07-01")
    assert eight_k is not None and eight_k.form is None
    assert parse_doc_name("not a doc name") is None


def test_candidates_include_neighbouring_years_closest_first() -> None:
    previous = replace(
        BASE_RECORD, accession="U-2023", report_date=date(2023, 1, 28), fiscal_year=2023
    )
    amendment = replace(BASE_RECORD, form="10-K/A", accession="U-2024A")
    far = replace(BASE_RECORD, accession="U-2020", fiscal_year=2020)
    doc = parse_doc_name("ULTABEAUTY_2023_10K")
    assert doc is not None
    candidates = candidate_filings(doc, [BASE_RECORD, amendment, far, previous])
    assert [record.accession for record in candidates] == ["U-2023", "U-2024"]


def test_dei_confirms() -> None:
    doc = parse_doc_name("ULTABEAUTY_2023_10K")
    assert doc is not None
    assert dei_confirms(doc, 2023, "FY") is True
    assert dei_confirms(doc, 2022, "FY") is False
    assert dei_confirms(doc, None, None) is None
