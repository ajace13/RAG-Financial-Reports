"""Data models shared across ingestion stages."""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_digits}/{document}"


@dataclass(frozen=True)
class FilingRecord:
    """One filing and the document we parse from it.

    For 10-K / 10-Q / DEF 14A that is EDGAR's primary document. For an 8-K earnings release
    it is the press-release exhibit (EX-99.1), which ingestion swaps in for the 8-K cover.
    `items` holds EDGAR's 8-K item codes (e.g. "2.02,9.01"); empty for other forms.
    """

    ticker: str
    cik: int
    company_name: str
    form: str
    accession: str
    filing_date: date
    report_date: date | None
    primary_document: str
    fiscal_year_end: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    items: str = ""

    @property
    def is_amendment(self) -> bool:
        return self.form.endswith("/A")

    @property
    def index_url(self) -> str:
        """EDGAR filing index page listing every document (and its type) in the filing."""
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession_digits=self.accession.replace("-", ""),
            document=f"{self.accession}-index.htm",
        )

    @property
    def source_url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik,
            accession_digits=self.accession.replace("-", ""),
            document=self.primary_document,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (dates as ISO strings)."""
        record = asdict(self)
        record["filing_date"] = self.filing_date.isoformat()
        record["report_date"] = self.report_date.isoformat() if self.report_date else None
        record["is_amendment"] = self.is_amendment
        record["source_url"] = self.source_url
        return record

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "FilingRecord":
        """Inverse of `to_dict`; derived keys (is_amendment, source_url) are ignored."""
        report_date = record.get("report_date")
        return cls(
            ticker=record["ticker"],
            cik=int(record["cik"]),
            company_name=record["company_name"],
            form=record["form"],
            accession=record["accession"],
            filing_date=date.fromisoformat(record["filing_date"]),
            report_date=date.fromisoformat(report_date) if report_date else None,
            primary_document=record["primary_document"],
            fiscal_year_end=record.get("fiscal_year_end"),
            fiscal_year=record.get("fiscal_year"),
            fiscal_period=record.get("fiscal_period"),
            items=record.get("items", ""),
        )
