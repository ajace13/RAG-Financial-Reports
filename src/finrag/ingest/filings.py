"""Discover periodic filings (10-K / 10-Q) for a company from EDGAR's submissions feed."""

import logging
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.fiscal import (
    estimate_prior_year_end,
    fiscal_year_label,
    is_plausible_prior_year_end,
    quarterly_fiscal_period,
)
from finrag.ingest.models import FilingRecord

logger = logging.getLogger(__name__)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"

# Transition reports (10-KT) are kept because they mark fiscal year-ends when a company
# changes its fiscal calendar, but they get no fiscal labels and are never selected by default.
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "10-KT/A"})
QUARTERLY_FORMS = frozenset({"10-Q", "10-Q/A"})
PERIODIC_FORMS = ANNUAL_FORMS | QUARTERLY_FORMS
# Older history pages are fetched back this far before `filed_since` so the 10-K preceding
# the earliest selected 10-Q is available for fiscal labelling.
HISTORY_MARGIN = timedelta(days=400)


# Companies missing from SEC's *current* ticker map (acquired or renamed). Ingestion logs the
# EDGAR company name for each CIK so these can be verified.
CIK_FALLBACKS = {
    "ATVI": 718877,  # Activision Blizzard (acquired by Microsoft, 2023)
    "XYZ": 1512673,  # Block, Inc. (formerly Square, ticker SQ)
    "SQ": 1512673,  # Block's former ticker
    "FL": 850209,  # Foot Locker
}


@dataclass(frozen=True)
class CompanyInfo:
    """Company identity and fiscal calendar as reported by EDGAR."""

    ticker: str
    cik: int
    name: str
    fiscal_year_end: str | None


def load_ticker_to_cik(client: EdgarClient) -> dict[str, int]:
    """Map current exchange tickers to CIKs using SEC's company_tickers.json."""
    payload = client.get_json(TICKER_MAP_URL)
    return {entry["ticker"].upper(): int(entry["cik_str"]) for entry in payload.values()}


def resolve_cik(client: EdgarClient, ticker: str) -> int | None:
    """CIK for a ticker: SEC's current ticker map first, then known delisted/renamed ones."""
    return {**CIK_FALLBACKS, **load_ticker_to_cik(client)}.get(ticker.upper())


def parse_date(value: str | None) -> date | None:
    """Parse an EDGAR ISO date, treating empty strings as missing."""
    return date.fromisoformat(value) if value else None


def parse_filing_table(
    table: Mapping[str, Sequence[Any]], company: CompanyInfo
) -> list[FilingRecord]:
    """Convert one columnar EDGAR filings table into unlabelled FilingRecords.

    EDGAR stores filings column-wise (`form`, `accessionNumber`, ... are parallel arrays).
    Non-periodic forms (8-K, 424B2, ...) are out of scope for this project and skipped.
    Fiscal labels need the company's whole 10-K history, so `assign_fiscal_labels` adds them.
    """
    records = []
    for index, form in enumerate(table.get("form", [])):
        if form not in PERIODIC_FORMS:
            continue
        records.append(
            FilingRecord(
                ticker=company.ticker,
                cik=company.cik,
                company_name=company.name,
                form=form,
                accession=table["accessionNumber"][index],
                filing_date=date.fromisoformat(table["filingDate"][index]),
                report_date=parse_date(table["reportDate"][index]),
                primary_document=table["primaryDocument"][index],
                fiscal_year_end=company.fiscal_year_end,
                fiscal_year=None,
                fiscal_period=None,
            )
        )
    return records


def fiscal_labels_for(
    record: FilingRecord, year_ends: Sequence[date], fiscal_year_end: str | None
) -> tuple[int | None, str | None]:
    """Fiscal (year, period) for one filing given the company's known fiscal year-end dates."""
    if record.report_date is None:
        return None, None
    if record.form in {"10-K", "10-K/A"}:
        return fiscal_year_label(record.report_date), "FY"
    if record.form not in QUARTERLY_FORMS:
        return None, None
    prior_candidates = [
        year_end
        for year_end in year_ends
        if is_plausible_prior_year_end(record.report_date, year_end)
    ]
    if prior_candidates:
        prior_year_end = max(prior_candidates)
    elif fiscal_year_end:
        prior_year_end = estimate_prior_year_end(record.report_date, fiscal_year_end)
    else:
        prior_year_end = None
    if prior_year_end is None:
        return None, None
    labels = quarterly_fiscal_period(record.report_date, prior_year_end)
    return labels if labels is not None else (None, None)


def assign_fiscal_labels(
    records: Sequence[FilingRecord], fiscal_year_end: str | None
) -> list[FilingRecord]:
    """Label every filing using year-ends taken from the company's own annual reports."""
    year_ends = sorted(
        {
            record.report_date
            for record in records
            if record.form in ANNUAL_FORMS and record.report_date
        }
    )
    labelled = []
    for record in records:
        fiscal_year, fiscal_period = fiscal_labels_for(record, year_ends, fiscal_year_end)
        labelled.append(replace(record, fiscal_year=fiscal_year, fiscal_period=fiscal_period))
    return labelled


def fetch_company_filings(
    client: EdgarClient, ticker: str, cik: int, filed_since: date | None = None
) -> tuple[CompanyInfo, list[FilingRecord]]:
    """Fetch and label every periodic filing for a company, following EDGAR's paginated history.

    The main submissions document inlines only the ~1000 most recent filings; older ones are
    in extra pages. Pages that end well before `filed_since` are not fetched.
    """
    submissions = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    company = CompanyInfo(
        ticker=ticker.upper(),
        cik=cik,
        name=submissions.get("name", ""),
        fiscal_year_end=submissions.get("fiscalYearEnd") or None,
    )
    records = parse_filing_table(submissions["filings"]["recent"], company)
    history_start = filed_since - HISTORY_MARGIN if filed_since is not None else None
    for page in submissions["filings"].get("files", []):
        page_end = parse_date(page.get("filingTo"))
        if history_start is not None and page_end is not None and page_end < history_start:
            continue
        page_table = client.get_json(SUBMISSIONS_PAGE_URL.format(name=page["name"]))
        records.extend(parse_filing_table(page_table, company))
    records = assign_fiscal_labels(records, company.fiscal_year_end)
    unlabelled = [
        record
        for record in records
        if record.fiscal_year is None and record.form not in {"10-KT", "10-KT/A"}
    ]
    if unlabelled:
        logger.warning(
            "%s: fiscal labels could not be derived for %d of %d periodic filings (%s)",
            company.ticker,
            len(unlabelled),
            len(records),
            ", ".join(f"{record.form} {record.report_date}" for record in unlabelled[:5]),
        )
    return company, records


def select_filings(
    records: Sequence[FilingRecord], forms: Collection[str], filed_since: date | None
) -> list[FilingRecord]:
    """Keep filings of the requested forms filed on/after `filed_since`, logging what was excluded.

    Amendments are excluded unless their form (e.g. '10-K/A') is requested explicitly.
    """
    excluded_by_form: Counter[str] = Counter()
    excluded_by_date = 0
    selected = []
    for record in records:
        if record.form not in forms:
            excluded_by_form[record.form] += 1
        elif filed_since is not None and record.filing_date < filed_since:
            excluded_by_date += 1
        else:
            selected.append(record)
    if excluded_by_form:
        logger.info("Excluded by form: %s", dict(excluded_by_form))
    if excluded_by_date:
        logger.info("Excluded %d filings filed before %s", excluded_by_date, filed_since)
    return sorted(selected, key=lambda record: record.filing_date)
