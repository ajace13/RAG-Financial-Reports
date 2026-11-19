"""FinanceBench open-source sample: question loading and mapping documents to EDGAR filings.

FinanceBench (Patronus AI) names each source document like `3M_2018_10K` or
`AMCOR_2023Q2_10Q`, using the company's own fiscal-year labels. We map those names to
tickers / CIKs and then to the matching filing in our manifest.
"""

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from finrag.ingest.models import FilingRecord

logger = logging.getLogger(__name__)

FINANCEBENCH_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/"
    "financebench_open_source.jsonl"
)

DOC_NAME_PATTERN = re.compile(
    r"^(?P<company_key>.+?)_(?P<year>\d{4})(?:Q(?P<quarter>[1-4]))?_(?P<doc_type>10K|10Q|8K|EARNINGS)",
    re.IGNORECASE,
)

# FinanceBench company key -> ticker used in our metadata.
COMPANY_TICKERS = {
    "3M": "MMM",
    "AES": "AES",
    "AMD": "AMD",
    "ACTIVISIONBLIZZARD": "ATVI",
    "ADOBE": "ADBE",
    "AMAZON": "AMZN",
    "AMCOR": "AMCR",
    "AMERICANEXPRESS": "AXP",
    "AMERICANWATERWORKS": "AWK",
    "BESTBUY": "BBY",
    "BLOCK": "XYZ",
    "BOEING": "BA",
    "CVSHEALTH": "CVS",
    "COCACOLA": "KO",
    "CORNING": "GLW",
    "COSTCO": "COST",
    "FOOTLOCKER": "FL",
    "GENERALMILLS": "GIS",
    "JPMORGAN": "JPM",
    "JOHNSON_JOHNSON": "JNJ",
    "KRAFTHEINZ": "KHC",
    "LOCKHEEDMARTIN": "LMT",
    "MGMRESORTS": "MGM",
    "MICROSOFT": "MSFT",
    "NETFLIX": "NFLX",
    "NIKE": "NKE",
    "PAYPAL": "PYPL",
    "PEPSICO": "PEP",
    "PFIZER": "PFE",
    "ULTABEAUTY": "ULTA",
    "VERIZON": "VZ",
    "WALMART": "WMT",
}

FORM_BY_DOC_TYPE = {"10K": "10-K", "10Q": "10-Q"}


@dataclass(frozen=True)
class FinanceBenchDoc:
    """A FinanceBench source document identified by its name."""

    doc_name: str
    company_key: str
    ticker: str | None
    year: int
    quarter: int | None
    doc_type: str

    @property
    def form(self) -> str | None:
        """EDGAR form for 10-K / 10-Q documents; None for 8-Ks and earnings releases."""
        return FORM_BY_DOC_TYPE.get(self.doc_type)

    @property
    def fiscal_period(self) -> str:
        return f"Q{self.quarter}" if self.quarter else "FY"


@dataclass(frozen=True)
class FinanceBenchQuestion:
    question_id: str
    question: str
    answer: str
    question_type: str
    doc: FinanceBenchDoc
    evidence_texts: tuple[str, ...]


def parse_doc_name(doc_name: str) -> FinanceBenchDoc | None:
    """Parse e.g. 'AMCOR_2023Q2_10Q'; returns None for names that do not follow the scheme."""
    match = DOC_NAME_PATTERN.match(doc_name)
    if match is None:
        return None
    company_key = match["company_key"].upper()
    return FinanceBenchDoc(
        doc_name=doc_name,
        company_key=company_key,
        ticker=COMPANY_TICKERS.get(company_key),
        year=int(match["year"]),
        quarter=int(match["quarter"]) if match["quarter"] else None,
        doc_type=match["doc_type"].upper(),
    )


def load_questions(path: Path) -> list[FinanceBenchQuestion]:
    """Load FinanceBench JSONL, logging (not silently dropping) rows with unparseable doc names."""
    questions = []
    unparseable = 0
    with path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue
            row = json.loads(line)
            doc = parse_doc_name(row["doc_name"])
            if doc is None:
                unparseable += 1
                logger.warning("Unparseable FinanceBench doc name: %s", row["doc_name"])
                continue
            questions.append(
                FinanceBenchQuestion(
                    question_id=row["financebench_id"],
                    question=row["question"],
                    answer=row["answer"],
                    question_type=row.get("question_type", ""),
                    doc=doc,
                    evidence_texts=tuple(item["evidence_text"] for item in row["evidence"]),
                )
            )
    logger.info("Loaded %d FinanceBench questions (%d unparseable)", len(questions), unparseable)
    return questions


def candidate_filings(doc: FinanceBenchDoc, records: Iterable[FilingRecord]) -> list[FilingRecord]:
    """Original (non-amended) filings that could be `doc`, best derived-label match first.

    Neighbouring fiscal years are included because fiscal-year naming conventions differ by
    company (start vs end year); the caller confirms the choice against the filing's dei facts.
    """
    candidates = [
        record
        for record in records
        if record.form == doc.form
        and record.fiscal_period == doc.fiscal_period
        and record.fiscal_year is not None
        and abs(record.fiscal_year - doc.year) <= 1
    ]
    return sorted(
        candidates,
        key=lambda record: (abs(record.fiscal_year - doc.year), record.filing_date),
    )


def dei_confirms(doc: FinanceBenchDoc, dei_year: int | None, dei_period: str | None) -> bool | None:
    """Whether a filing's dei fiscal facts match `doc`; None if the filing has no dei facts."""
    if dei_year is None:
        return None
    period_matches = dei_period is None or dei_period == doc.fiscal_period
    return dei_year == doc.year and period_matches
