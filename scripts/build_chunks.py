"""Parse every filing in the manifest into chunks.

Writes data/processed/chunks.jsonl (one chunk per line) and
data/processed/parse_report.jsonl (one diagnostics record per filing).
"""

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from finrag.chunking import ChunkingConfig
from finrag.config import load_settings
from finrag.ingest.models import FilingRecord
from finrag.ingest.storage import raw_document_path, read_manifest
from finrag.logging_utils import configure_logging
from finrag.parsing.pipeline import FilingParseReport, chunk_filing

logger = logging.getLogger("build_chunks")


def process_filing(record_dict: dict, raw_dir: str) -> tuple[list[dict], dict]:
    """Worker: parse one filing. Errors are captured in the report, not raised."""
    from pathlib import Path

    record = FilingRecord.from_dict(record_dict)
    try:
        html = raw_document_path(Path(raw_dir), record).read_bytes()
        chunks, report = chunk_filing(record, html, ChunkingConfig())
    except Exception as error:  # noqa: BLE001 - one bad filing must not abort the batch
        report = FilingParseReport(
            accession=record.accession, ticker=record.ticker, form=record.form, error=repr(error)
        )
        return [], report.to_dict()
    return [chunk.to_dict() for chunk in chunks], report.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    return parser.parse_args()


def main() -> int:
    """Chunk all manifest filings in parallel and log a corpus-level summary."""
    configure_logging()
    args = parse_args()
    settings = load_settings()
    raw_dir = settings.data_dir / "raw"
    processed_dir = settings.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(raw_dir)
    logger.info("Chunking %d filings with %d workers", len(manifest), args.workers)

    chunk_count = 0
    reports: list[dict] = []
    chunks_path = processed_dir / "chunks.jsonl"
    with (
        chunks_path.open("w", encoding="utf-8") as chunks_file,
        ProcessPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures = [
            executor.submit(process_filing, record.to_dict(), str(raw_dir))
            for record in manifest.values()
        ]
        for future in as_completed(futures):
            chunk_dicts, report = future.result()
            for chunk in chunk_dicts:
                chunks_file.write(json.dumps(chunk) + "\n")
            chunk_count += len(chunk_dicts)
            reports.append(report)
            if len(reports) % 20 == 0:
                logger.info("Processed %d/%d filings", len(reports), len(manifest))

    reports.sort(key=lambda report: (report["ticker"], report["accession"]))
    with (processed_dir / "parse_report.jsonl").open("w", encoding="utf-8") as report_file:
        for report in reports:
            report_file.write(json.dumps(report) + "\n")

    failed = [report for report in reports if report["error"]]
    for report in failed:
        logger.error("Failed %s %s: %s", report["ticker"], report["accession"], report["error"])
    label_sources = Counter(report["label_source"] for report in reports if not report["error"])
    relabelled = [
        report
        for report in reports
        if report["label_source"] == "dei"
        and (report["fiscal_year"], report["fiscal_period"])
        != (report["derived_fiscal_year"], report["derived_fiscal_period"])
    ]
    for report in relabelled:
        logger.info(
            "dei labels differ from derived for %s %s %s: FY%s %s (derived FY%s %s)",
            report["ticker"],
            report["form"],
            report["accession"],
            report["fiscal_year"],
            report["fiscal_period"],
            report["derived_fiscal_year"],
            report["derived_fiscal_period"],
        )
    unsectioned = sum(report["unsectioned_chunks"] for report in reports)
    logger.info(
        "Done: %d filings ok, %d failed; %d chunks (%.1f%% in unidentified sections); "
        "fiscal label sources %s; %d tiny chunks dropped",
        len(reports) - len(failed),
        len(failed),
        chunk_count,
        100 * unsectioned / max(chunk_count, 1),
        dict(label_sources),
        sum(report["dropped_tiny_chunks"] for report in reports),
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
