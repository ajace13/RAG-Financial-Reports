"""Download the FinanceBench open sample and every 10-K / 10-Q filing it cites.

Writes data/eval/financebench_open_source.jsonl, data/eval/financebench_doc_map.json
(doc_name -> accession) and adds the filings to data/raw/ and the manifest.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import requests

from finrag.config import load_settings
from finrag.evaluation.financebench import (
    CIK_FALLBACKS,
    FINANCEBENCH_URL,
    FinanceBenchDoc,
    candidate_filings,
    dei_confirms,
    load_questions,
)
from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.filings import fetch_company_filings, load_ticker_to_cik
from finrag.ingest.models import FilingRecord
from finrag.ingest.storage import download_filings, merge_into_manifest, raw_document_path
from finrag.logging_utils import configure_logging
from finrag.parsing.dei import fiscal_labels_from_dei, read_dei_facts

logger = logging.getLogger("ingest_financebench")


def resolve_document(
    client: EdgarClient, doc: FinanceBenchDoc, records: list[FilingRecord], raw_dir: Path
) -> tuple[FilingRecord | None, str]:
    """Pick the filing a FinanceBench document refers to; returns (filing, method).

    Candidates are downloaded in order of derived-label closeness and confirmed against
    their own iXBRL dei facts. Filings without dei facts (roughly pre-2019) fall back to an
    exact derived-label match.
    """
    derived_fallback = None
    for record in candidate_filings(doc, records):
        summary = download_filings(client, [record], raw_dir)
        if summary.failed:
            continue
        dei_facts = read_dei_facts(raw_document_path(raw_dir, record).read_bytes())
        dei_year, dei_period = fiscal_labels_from_dei(dei_facts)
        verdict = dei_confirms(doc, dei_year, dei_period)
        if verdict is True:
            return replace(record, fiscal_year=dei_year, fiscal_period=doc.fiscal_period), "dei"
        if verdict is None and record.fiscal_year == doc.year and derived_fallback is None:
            derived_fallback = record
    if derived_fallback is not None:
        return derived_fallback, "derived"
    return None, "unresolved"


def main() -> int:
    """Resolve FinanceBench documents to EDGAR filings and download them. Returns an exit code."""
    configure_logging()
    settings = load_settings()
    client = EdgarClient(settings.sec_user_agent or "")
    eval_dir = settings.data_dir / "eval"
    raw_dir = settings.data_dir / "raw"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = eval_dir / "financebench_open_source.jsonl"
    if not dataset_path.exists():
        logger.info("Downloading FinanceBench open sample")
        # Plain request: the SEC User-Agent (contact email) is only meant for sec.gov.
        response = requests.get(FINANCEBENCH_URL, timeout=60)
        response.raise_for_status()
        dataset_path.write_bytes(response.content)
    questions = load_questions(dataset_path)

    docs = {question.doc.doc_name: question.doc for question in questions}
    periodic_docs = [doc for doc in docs.values() if doc.form is not None]
    logger.info(
        "%d distinct documents; %d are 10-K/10-Q, %d skipped (8-K / earnings releases)",
        len(docs),
        len(periodic_docs),
        len(docs) - len(periodic_docs),
    )

    docs_by_ticker: dict[str, list[FinanceBenchDoc]] = defaultdict(list)
    for doc in periodic_docs:
        if doc.ticker is None:
            logger.error("No ticker mapping for FinanceBench company %s", doc.company_key)
            continue
        docs_by_ticker[doc.ticker].append(doc)

    ticker_to_cik = {**CIK_FALLBACKS, **load_ticker_to_cik(client)}
    doc_map: dict[str, str] = {}
    unresolved: list[str] = []
    resolution_methods: Counter[str] = Counter()
    for ticker, ticker_docs in sorted(docs_by_ticker.items()):
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            logger.error("No CIK for %s; skipping %d documents", ticker, len(ticker_docs))
            unresolved.extend(doc.doc_name for doc in ticker_docs)
            continue
        company, records = fetch_company_filings(client, ticker, cik)
        logger.info("%s -> CIK %d (%s), FYE %s", ticker, cik, company.name, company.fiscal_year_end)
        matched = []
        for doc in ticker_docs:
            record, method = resolve_document(client, doc, records, raw_dir)
            if record is None:
                logger.warning("No filing matches %s", doc.doc_name)
                unresolved.append(doc.doc_name)
                continue
            resolution_methods[method] += 1
            logger.info(
                "%s -> %s %s (period end %s, via %s)",
                doc.doc_name,
                record.form,
                record.accession,
                record.report_date,
                method,
            )
            doc_map[doc.doc_name] = record.accession
            matched.append(record)
        merge_into_manifest(raw_dir, matched)

    logger.info("Resolution methods: %s", dict(resolution_methods))
    doc_map_path = eval_dir / "financebench_doc_map.json"
    doc_map_path.write_text(json.dumps(doc_map, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "Resolved %d of %d periodic documents; unresolved: %s",
        len(doc_map),
        len(periodic_docs),
        sorted(unresolved) or "none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
