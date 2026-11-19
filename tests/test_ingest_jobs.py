"""Incremental ingestion planning and the background job manager (no network, no GPU)."""

import threading
import time
from datetime import date
from typing import Any

from finrag.index.facets import indexed_accessions
from finrag.ingest.jobs import IngestJobManager
from finrag.ingest.models import FilingRecord
from finrag.ingest.service import (
    IngestProgress,
    IngestRequest,
    IngestResult,
    filed_since_for,
    plan_new_filings,
)


def record(accession: str, document: str = "doc.htm") -> FilingRecord:
    return FilingRecord(
        ticker="NVDA",
        cik=1045810,
        company_name="NVIDIA CORP",
        form="10-K",
        accession=accession,
        filing_date=date(2024, 2, 21),
        report_date=date(2024, 1, 28),
        primary_document=document,
        fiscal_year_end="0128",
        fiscal_year=2024,
        fiscal_period="FY",
    )


def test_filed_since_recent_vs_full_history() -> None:
    today = date(2026, 11, 19)
    assert filed_since_for(IngestRequest("NVDA"), today) == date(2021, 11, 19)
    assert filed_since_for(IngestRequest("NVDA", full_history=True), today) is None


def test_plan_skips_indexed_and_plain_text_filings() -> None:
    records = [
        record("A-1"),
        record("A-2"),
        record("A-3", "old-filing.txt"),
        record("A-4", "x.HTM"),
    ]
    new, already, non_html = plan_new_filings(records, already_indexed={"A-2"})
    assert [r.accession for r in new] == ["A-1", "A-4"]
    assert (already, non_html) == (1, 1)


def test_scope_label() -> None:
    assert IngestRequest("NVDA").scope_label == "last 5 years"
    assert IngestRequest("NVDA", full_history=True).scope_label == "full EDGAR history"


class FakeES:
    def search(self, **kwargs: Any) -> dict:
        assert kwargs["query"] == {"bool": {"filter": [{"term": {"ticker": "MSFT"}}]}}
        return {"aggregations": {"accessions": {"buckets": [{"key": "A-1"}, {"key": "A-2"}]}}}


def test_indexed_accessions() -> None:
    assert indexed_accessions(FakeES(), "index", "MSFT") == {"A-1", "A-2"}


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def test_jobs_run_in_order_and_report_progress_results_and_errors() -> None:
    order: list[str] = []

    def run(request: IngestRequest, report, gpu_lock) -> IngestResult:
        order.append(request.ticker)
        report(IngestProgress("Embedding passages", 5, 10))
        if request.ticker == "BAD":
            raise ValueError("Ticker BAD was not found on SEC EDGAR.")
        return IngestResult(ticker=request.ticker, filings_added=3, chunks_added=900)

    manager = IngestJobManager(run)
    manager.submit(IngestRequest("NVDA"))
    manager.submit(IngestRequest("BAD"))
    wait_until(lambda: manager.finished_count() == 2)
    jobs = {job.request.ticker: job for job in manager.snapshot()}
    assert order == ["NVDA", "BAD"]
    assert jobs["NVDA"].status == "done" and jobs["NVDA"].result.chunks_added == 900
    assert jobs["NVDA"].progress.fraction == 0.5
    assert jobs["BAD"].status == "failed" and "not found" in jobs["BAD"].error


def test_duplicate_submission_returns_existing_job() -> None:
    release = threading.Event()

    def run(request: IngestRequest, report, gpu_lock) -> IngestResult:
        release.wait(5)
        return IngestResult(ticker=request.ticker)

    manager = IngestJobManager(run)
    first = manager.submit(IngestRequest("NVDA"))
    second = manager.submit(IngestRequest("NVDA"))
    full = manager.submit(IngestRequest("NVDA", full_history=True))
    assert first.job_id == second.job_id != full.job_id
    release.set()
    wait_until(lambda: manager.finished_count() == 2)


def test_gpu_stage_waits_while_a_question_holds_the_lock() -> None:
    entered_gpu_stage = threading.Event()

    def run(request: IngestRequest, report, gpu_lock) -> IngestResult:
        with gpu_lock():
            entered_gpu_stage.set()
        return IngestResult(ticker=request.ticker)

    manager = IngestJobManager(run)
    manager.gpu_lock.acquire()  # a question is being answered
    manager.submit(IngestRequest("NVDA"))
    assert not entered_gpu_stage.wait(0.3)
    manager.gpu_lock.release()
    assert entered_gpu_stage.wait(5)
    wait_until(lambda: manager.finished_count() == 1)
