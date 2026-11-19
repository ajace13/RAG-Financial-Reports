"""Background ingestion jobs for the app: one worker thread, one job at a time.

`gpu_lock` is shared with question answering. The app holds it while answering; ingestion
holds it only for its embed + index stage. Downloads and parsing therefore overlap with
questions, but the GPU-heavy steps never run at the same time.
"""

import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace

from finrag.ingest.service import IngestProgress, IngestRequest, IngestResult

logger = logging.getLogger(__name__)

RunIngest = Callable[
    [IngestRequest, Callable[[IngestProgress], None], Callable[[], AbstractContextManager]],
    IngestResult,
]
ACTIVE_STATUSES = ("queued", "running")


@dataclass
class IngestJob:
    job_id: int
    request: IngestRequest
    status: str = "queued"  # queued | running | done | failed
    progress: IngestProgress = field(default_factory=IngestProgress)
    result: IngestResult | None = None
    error: str | None = None
    submitted_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class IngestJobManager:
    def __init__(self, run_ingest: RunIngest) -> None:
        self.run_ingest = run_ingest
        self.gpu_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._jobs: list[IngestJob] = []
        self._queue: queue.Queue[IngestJob] = queue.Queue()
        self._ids = itertools.count(1)
        self._worker: threading.Thread | None = None

    def submit(self, request: IngestRequest) -> IngestJob:
        """Queue a job; an identical queued/running job is returned instead of duplicated."""
        with self._state_lock:
            for job in self._jobs:
                if job.request == request and job.status in ACTIVE_STATUSES:
                    return replace(job)
            job = IngestJob(job_id=next(self._ids), request=request)
            self._jobs.append(job)
            self._queue.put(job)
            if self._worker is None:
                self._worker = threading.Thread(target=self._work, name="ingest", daemon=True)
                self._worker.start()
            return replace(job)

    def snapshot(self) -> list[IngestJob]:
        """Copies of all jobs, newest first (safe to read from the UI thread)."""
        with self._state_lock:
            return [replace(job, progress=replace(job.progress)) for job in reversed(self._jobs)]

    def active_job(self) -> IngestJob | None:
        return next((job for job in self.snapshot() if job.status == "running"), None)

    def finished_count(self) -> int:
        with self._state_lock:
            return sum(1 for job in self._jobs if job.status not in ACTIVE_STATUSES)

    def _update(self, job: IngestJob, **changes: object) -> None:
        with self._state_lock:
            for name, value in changes.items():
                setattr(job, name, value)

    def _work(self) -> None:
        # Blocks forever between jobs: a daemon thread costs nothing while idle, and never
        # exiting avoids a race where a job is queued just as an idle worker shuts down.
        while True:
            job = self._queue.get()
            self._update(job, status="running")

            def report(progress: IngestProgress, job: IngestJob = job) -> None:
                self._update(job, progress=progress)

            try:
                result = self.run_ingest(job.request, report, lambda: self.gpu_lock)
            except Exception as error:  # noqa: BLE001 - surface every failure in the UI
                logger.exception("Ingestion failed for %s", job.request.ticker)
                self._update(job, status="failed", error=str(error), finished_at=time.time())
            else:
                self._update(job, status="done", result=result, finished_at=time.time())
            finally:
                self._queue.task_done()
