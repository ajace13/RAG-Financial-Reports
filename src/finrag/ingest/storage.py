"""On-disk layout for raw filings and the filing manifest."""

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import requests

from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.models import FilingRecord

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.jsonl"


@dataclass
class DownloadSummary:
    """Outcome of a download batch. `succeeded` includes files that were already on disk."""

    downloaded: int = 0
    skipped_existing: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    succeeded: list[FilingRecord] = field(default_factory=list)


def raw_document_path(raw_dir: Path, record: FilingRecord) -> Path:
    """Where a filing's primary document is stored: <raw_dir>/<TICKER>/<accession><suffix>."""
    suffix = PurePosixPath(record.primary_document).suffix or ".htm"
    return raw_dir / record.ticker / f"{record.accession}{suffix}"


def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_FILENAME


def read_manifest(raw_dir: Path) -> dict[str, FilingRecord]:
    """Load the manifest keyed by accession number; missing manifest means no filings yet."""
    path = manifest_path(raw_dir)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as manifest_file:
        records = [
            FilingRecord.from_dict(json.loads(line)) for line in manifest_file if line.strip()
        ]
    return {record.accession: record for record in records}


def write_manifest(raw_dir: Path, records: Mapping[str, FilingRecord]) -> None:
    """Rewrite the manifest atomically, sorted by ticker then filing date."""
    path = manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda record: (record.ticker, record.filing_date))
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as manifest_file:
        for record in ordered:
            manifest_file.write(json.dumps(record.to_dict()) + "\n")
    temporary_path.replace(path)


def merge_into_manifest(raw_dir: Path, records: Iterable[FilingRecord]) -> int:
    """Add or update records in the manifest; returns the manifest size afterwards."""
    manifest = read_manifest(raw_dir)
    for record in records:
        manifest[record.accession] = record
    write_manifest(raw_dir, manifest)
    return len(manifest)


def download_filings(
    client: EdgarClient, records: Iterable[FilingRecord], raw_dir: Path
) -> DownloadSummary:
    """Download each filing's primary document, skipping files already on disk.

    Failures are recorded and logged rather than raised so one bad filing does not abort
    a batch.
    """
    summary = DownloadSummary()
    for record in records:
        path = raw_document_path(raw_dir, record)
        if path.exists() and path.stat().st_size > 0:
            summary.skipped_existing += 1
            summary.succeeded.append(record)
            continue
        try:
            content = client.get_bytes(record.source_url)
        except requests.RequestException as error:
            logger.error("Download failed for %s %s: %s", record.ticker, record.accession, error)
            summary.failed.append((record.accession, str(error)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".part")
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
        summary.downloaded += 1
        summary.succeeded.append(record)
        logger.info(
            "Downloaded %s %s %s FY%s %s (%.1f MB)",
            record.ticker,
            record.form,
            record.accession,
            record.fiscal_year,
            record.fiscal_period,
            len(content) / 1e6,
        )
    return summary
