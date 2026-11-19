"""Discover a company's filings (10-K, 10-Q, earnings-release 8-Ks, proxies) from EDGAR."""

import logging
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

from bs4 import BeautifulSoup

from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.fiscal import (
    earnings_release_period,
    estimate_prior_year_end,
    fiscal_year_label,
    is_plausible_prior_year_end,
    latest_year_end_before,
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
# Optional document types: earnings releases arrive as 8-Ks reporting Item 2.02 (results of
# operations) with the press release attached as Exhibit 99.1; proxies are DEF 14A.
EARNINGS_RELEASE_FORM = "8-K"
EARNINGS_RELEASE_ITEM = "2.02"
PROXY_FORM = "DEF 14A"
SUPPLEMENTAL_FORMS = frozenset({EARNINGS_RELEASE_FORM, PROXY_FORM})
EXHIBIT_99_PATTERN = re.compile(r"^EX-99(?:\.0?1)?$", re.IGNORECASE)
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
    Periodic reports, 8-Ks and proxies are kept; other forms (424B2, S-8, ...) are out of
    scope. Fiscal labels need the company's whole 10-K history, so `assign_fiscal_labels`
    adds them afterwards.
    """
    records = []
    items_column = table.get("items") or []
    for index, form in enumerate(table.get("form", [])):
        if form not in PERIODIC_FORMS and form not in SUPPLEMENTAL_FORMS:
            continue
        items = items_column[index] if index < len(items_column) else ""
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
                items=items or "",
            )
        )
    return records


def is_earnings_release(record: FilingRecord) -> bool:
    """An 8-K that reports results of operations (Item 2.02)."""
    item_codes = {code.strip() for code in record.items.split(",")}
    return record.form == EARNINGS_RELEASE_FORM and EARNINGS_RELEASE_ITEM in item_codes


def find_exhibit_document(index_html: str) -> str | None:
    """Document name of a filing's press-release exhibit (EX-99.1, else EX-99) from its index page.

    The index page has a table with columns Seq | Description | Document | Type | Size.
    """
    soup = BeautifulSoup(index_html, "lxml")
    fallback = None
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) < 4 or not EXHIBIT_99_PATTERN.match(cells[3]):
            continue
        document = cells[2].split()[0] if cells[2] else ""
        if not document:
            continue
        if cells[3].upper() in {"EX-99.1", "EX-99.01"}:
            return document
        fallback = fallback or document
    return fallback


def quarterly_labels(
    record: FilingRecord, year_ends: Sequence[date], fiscal_year_end: str | None
) -> tuple[int | None, str | None]:
    """Fiscal labels for a 10-Q, placed relative to the preceding fiscal year-end."""
    if record.report_date is None:
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


def supplemental_labels(
    record: FilingRecord, year_ends: Sequence[date], fiscal_year_end: str | None
) -> tuple[int | None, str | None]:
    """Fiscal labels for earnings releases and proxies, which carry no period of their own.

    An earnings release reports the quarter that closed just before it was filed; a proxy
    statement covers the fiscal year that ended before it was filed (executive pay etc.).
    """
    prior_year_end = latest_year_end_before(record.filing_date, year_ends, fiscal_year_end)
    if prior_year_end is None:
        return None, None
    if record.form == PROXY_FORM:
        return fiscal_year_label(prior_year_end), "FY"
    labels = earnings_release_period(record.filing_date, prior_year_end)
    return labels if labels is not None else (None, None)


def fiscal_labels_for(
    record: FilingRecord, year_ends: Sequence[date], fiscal_year_end: str | None
) -> tuple[int | None, str | None]:
    """Fiscal (year, period) for one filing given the company's known fiscal year-end dates."""
    if record.form in {"10-K", "10-K/A"}:
        return (fiscal_year_label(record.report_date), "FY") if record.report_date else (None, None)
    if record.form in QUARTERLY_FORMS:
        return quarterly_labels(record, year_ends, fiscal_year_end)
    if record.form == PROXY_FORM or is_earnings_release(record):
        return supplemental_labels(record, year_ends, fiscal_year_end)
    return None, None


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
    labelable = [record for record in records if expects_fiscal_labels(record)]
    unlabelled = [record for record in labelable if record.fiscal_year is None]
    if unlabelled:
        logger.warning(
            "%s: fiscal labels could not be derived for %d of %d filings (%s)",
            company.ticker,
            len(unlabelled),
            len(labelable),
            ", ".join(f"{record.form} {record.report_date}" for record in unlabelled[:5]),
        )
    return company, records


def expects_fiscal_labels(record: FilingRecord) -> bool:
    """Filings we label: 10-K/10-Q (not transition reports), proxies, earnings releases."""
    if record.form in PERIODIC_FORMS:
        return record.form not in {"10-KT", "10-KT/A"}
    return record.form == PROXY_FORM or is_earnings_release(record)


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
