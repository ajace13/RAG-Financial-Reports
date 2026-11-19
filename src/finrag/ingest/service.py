"""Incremental end-to-end ingestion of one company: EDGAR -> chunks -> embeddings -> index.

Used by the app's background jobs and by scripts/add_ticker.py. Filings already in the
index are skipped, so extending a ticker from "last 5 years" to "full history" only fetches
the older filings.

Stages and where they run:
- discover / download (network) and parse / chunk (CPU) run without any lock, so they can
  overlap with questions being answered;
- embed + index take `gpu_lock` (the same lock the app holds while answering), unload the
  local Ollama model first, and embed in fp16 on the GPU, so ingestion and generation never
  compete for a small GPU's memory.
"""

import contextlib
import gc
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import PurePosixPath

import requests
import torch

from finrag.chunking import ChunkingConfig
from finrag.config import Settings
from finrag.embedding import Embedder, load_embedding_cache, save_embedding_cache
from finrag.generate.ollama import unload_model
from finrag.index.client import make_client
from finrag.index.facets import indexed_accessions
from finrag.index.indexer import append_chunks, bulk_index
from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.filings import (
    EARNINGS_RELEASE_FORM,
    PROXY_FORM,
    fetch_company_filings,
    find_exhibit_document,
    is_earnings_release,
    resolve_cik,
    select_filings,
)
from finrag.ingest.models import FilingRecord
from finrag.ingest.storage import download_filings, merge_into_manifest, raw_document_path
from finrag.parsing.pipeline import chunk_filing

logger = logging.getLogger(__name__)

RECENT_YEARS = 5
INGEST_FORMS = ("10-K", "10-Q")
# Pre-2001 filings are mostly plain text without table markup; they are skipped (and counted).
HTML_SUFFIXES = (".htm", ".html")
EMBED_BATCH = 256


class IngestError(RuntimeError):
    """Ingestion could not proceed (unknown ticker, missing SEC User-Agent, ...)."""


@dataclass(frozen=True)
class IngestRequest:
    """What to fetch for one company. Earnings releases and proxies are opt-in."""

    ticker: str
    full_history: bool = False
    include_earnings_releases: bool = False
    include_proxies: bool = False

    @property
    def forms(self) -> tuple[str, ...]:
        extras = (EARNINGS_RELEASE_FORM,) if self.include_earnings_releases else ()
        extras += (PROXY_FORM,) if self.include_proxies else ()
        return INGEST_FORMS + extras

    @property
    def scope_label(self) -> str:
        period = "full EDGAR history" if self.full_history else f"last {RECENT_YEARS} years"
        extras = [
            label
            for label, wanted in (
                ("earnings releases", self.include_earnings_releases),
                ("proxies", self.include_proxies),
            )
            if wanted
        ]
        return f"{period} + {' + '.join(extras)}" if extras else period


@dataclass
class IngestProgress:
    stage: str = "queued"
    done: int = 0
    total: int = 0
    detail: str = ""

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


@dataclass
class IngestResult:
    ticker: str
    company_name: str = ""
    filings_found: int = 0
    already_indexed: int = 0
    skipped_non_html: int = 0
    missing_exhibits: int = 0
    filings_added: int = 0
    chunks_added: int = 0
    failed: list[str] = field(default_factory=list)


ProgressCallback = Callable[[IngestProgress], None]
GpuLockFactory = Callable[[], AbstractContextManager]


def filed_since_for(request: IngestRequest, today: date) -> date | None:
    """Filing-date cutoff: None means the company's entire EDGAR history."""
    if request.full_history:
        return None
    return today - timedelta(days=round(365.25 * RECENT_YEARS))


def plan_new_filings(
    records: list[FilingRecord], already_indexed: set[str]
) -> tuple[list[FilingRecord], int, int]:
    """Filings to ingest, plus counts skipped because indexed already / not HTML."""
    new, skipped_indexed, skipped_non_html = [], 0, 0
    for record in records:
        if record.accession in already_indexed:
            skipped_indexed += 1
        elif PurePosixPath(record.primary_document).suffix.lower() not in HTML_SUFFIXES:
            skipped_non_html += 1
        else:
            new.append(record)
    return new, skipped_indexed, skipped_non_html


def select_requested_filings(
    records: list[FilingRecord], request: IngestRequest, filed_since: date | None
) -> list[FilingRecord]:
    """Requested forms in the time window; 8-Ks only when they are earnings releases."""
    selected = select_filings(records, request.forms, filed_since)
    kept = [
        record
        for record in selected
        if record.form != EARNINGS_RELEASE_FORM or is_earnings_release(record)
    ]
    if len(kept) < len(selected):
        logger.info("Skipped %d 8-Ks that are not earnings releases", len(selected) - len(kept))
    return kept


def attach_release_exhibits(
    client: EdgarClient, records: list[FilingRecord], report: ProgressCallback
) -> tuple[list[FilingRecord], int]:
    """Point each earnings-release 8-K at its press-release exhibit (EX-99.1).

    The 8-K's own primary document is a one-page cover; the numbers are in the exhibit.
    Returns the updated records and how many 8-Ks had no findable exhibit (dropped).
    """
    resolved, missing = [], 0
    releases = [record for record in records if record.form == EARNINGS_RELEASE_FORM]
    for position, record in enumerate(records, start=1):
        if record.form != EARNINGS_RELEASE_FORM:
            resolved.append(record)
            continue
        report(IngestProgress("Finding press-release exhibits", position, len(releases)))
        try:
            index_html = client.get_bytes(record.index_url).decode("utf-8", errors="replace")
        except requests.RequestException as error:
            logger.warning("Index page failed for %s: %s", record.accession, error)
            missing += 1
            continue
        document = find_exhibit_document(index_html)
        if document is None:
            logger.warning("No EX-99 exhibit in 8-K %s", record.accession)
            missing += 1
            continue
        resolved.append(replace(record, primary_document=document))
    return resolved, missing


def batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def ingest_company(
    settings: Settings,
    request: IngestRequest,
    report: ProgressCallback,
    gpu_lock: GpuLockFactory = contextlib.nullcontext,
    today: date | None = None,
) -> IngestResult:
    """Fetch, parse, embed and index a company's new filings of the requested types."""
    ticker = request.ticker.strip().upper()
    result = IngestResult(ticker=ticker)
    if not settings.sec_user_agent:
        raise IngestError("SEC_USER_AGENT is not set in .env (SEC requires contact info).")
    client = EdgarClient(settings.sec_user_agent)

    report(IngestProgress("Looking up company on EDGAR", detail=ticker))
    cik = resolve_cik(client, ticker)
    if cik is None:
        raise IngestError(f"Ticker {ticker} was not found on SEC EDGAR.")
    filed_since = filed_since_for(request, today or date.today())
    company, records = fetch_company_filings(client, ticker, cik, filed_since)
    result.company_name = company.name
    selected = select_requested_filings(records, request, filed_since)
    result.filings_found = len(selected)

    es = make_client(settings)
    already = indexed_accessions(es, settings.es_index, ticker)
    not_indexed = [record for record in selected if record.accession not in already]
    result.already_indexed = len(selected) - len(not_indexed)
    with_documents, result.missing_exhibits = attach_release_exhibits(client, not_indexed, report)
    new_records, _, result.skipped_non_html = plan_new_filings(with_documents, set())
    logger.info(
        "%s (%s): %d filings in scope, %d already indexed, %d without exhibit, "
        "%d non-HTML skipped, %d new",
        ticker,
        company.name,
        len(selected),
        result.already_indexed,
        result.missing_exhibits,
        result.skipped_non_html,
        len(new_records),
    )
    if not new_records:
        return result

    raw_dir = settings.data_dir / "raw"
    downloaded: list[FilingRecord] = []
    for position, record in enumerate(new_records, start=1):
        report(IngestProgress("Downloading filings", position - 1, len(new_records), record.form))
        summary = download_filings(client, [record], raw_dir)
        downloaded.extend(summary.succeeded)
        result.failed.extend(accession for accession, _ in summary.failed)
    merge_into_manifest(raw_dir, downloaded)

    chunks: list[dict] = []
    for position, record in enumerate(downloaded, start=1):
        label = f"{record.form} FY{record.fiscal_year} {record.fiscal_period or ''}".strip()
        report(IngestProgress("Parsing filings", position - 1, len(downloaded), label))
        try:
            html = raw_document_path(raw_dir, record).read_bytes()
            filing_chunks, _ = chunk_filing(record, html, ChunkingConfig())
        except Exception as error:  # noqa: BLE001 - one malformed filing must not stop the job
            logger.error("Parsing failed for %s %s: %r", ticker, record.accession, error)
            result.failed.append(record.accession)
            continue
        chunks.extend(chunk.to_dict() for chunk in filing_chunks)
        result.filings_added += 1
    if not chunks:
        return result

    report(IngestProgress("Waiting for any running question to finish", 0, len(chunks)))
    with gpu_lock():
        if settings.ollama_model and "localhost" in settings.ollama_url:
            unload_model(settings.ollama_url, settings.ollama_model)
        embedder = Embedder(settings.embedding_model, half_precision=True)
        vectors = []
        for batch in batched(chunks, EMBED_BATCH):
            report(IngestProgress("Embedding passages", len(vectors), len(chunks)))
            vectors.extend(embedder.embed_documents([chunk["search_text"] for chunk in batch]))
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        report(IngestProgress("Indexing", len(chunks), len(chunks)))
        documents = (
            {**chunk, "embedding": vector.tolist()}
            for chunk, vector in zip(chunks, vectors, strict=True)
        )
        indexed, failed = bulk_index(es, settings.es_index, documents)
        es.indices.refresh(index=settings.es_index)
    if failed:
        logger.error("%d chunks failed to index for %s", failed, ticker)

    processed_dir = settings.data_dir / "processed"
    append_chunks(processed_dir / "chunks.jsonl", chunks)
    cache_dir = processed_dir / "embeddings"
    cache = load_embedding_cache(cache_dir, settings.embedding_model)
    cache.update((chunk["chunk_id"], vector) for chunk, vector in zip(chunks, vectors, strict=True))
    save_embedding_cache(cache_dir, settings.embedding_model, cache)
    result.chunks_added = indexed
    report(IngestProgress("Done", len(chunks), len(chunks)))
    return result
