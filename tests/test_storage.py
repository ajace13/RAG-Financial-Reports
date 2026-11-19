from datetime import date
from pathlib import Path

import requests

from finrag.ingest.models import FilingRecord
from finrag.ingest.storage import (
    download_filings,
    merge_into_manifest,
    raw_document_path,
    read_manifest,
)


def make_record(accession: str, document: str = "doc.htm") -> FilingRecord:
    return FilingRecord(
        ticker="MSFT",
        cik=789019,
        company_name="MICROSOFT CORP",
        form="10-K",
        accession=accession,
        filing_date=date(2024, 7, 30),
        report_date=date(2024, 6, 30),
        primary_document=document,
        fiscal_year_end="0630",
        fiscal_year=2024,
        fiscal_period="FY",
    )


class FakeClient:
    def __init__(self, failing_urls: set[str]) -> None:
        self.failing_urls = failing_urls
        self.fetched: list[str] = []

    def get_bytes(self, url: str) -> bytes:
        self.fetched.append(url)
        if url in self.failing_urls:
            raise requests.HTTPError("404")
        return b"<html>filing</html>"


def test_manifest_round_trip_and_merge(tmp_path: Path) -> None:
    first, second = make_record("0001-24-000001"), make_record("0001-24-000002")
    assert merge_into_manifest(tmp_path, [first]) == 1
    assert merge_into_manifest(tmp_path, [first, second]) == 2
    manifest = read_manifest(tmp_path)
    assert manifest["0001-24-000002"] == second


def test_download_skips_existing_and_records_failures(tmp_path: Path) -> None:
    existing, fresh, broken = (
        make_record("0001-24-000001"),
        make_record("0001-24-000002"),
        make_record("0001-24-000003"),
    )
    existing_path = raw_document_path(tmp_path, existing)
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"already here")

    client = FakeClient(failing_urls={broken.source_url})
    summary = download_filings(client, [existing, fresh, broken], tmp_path)

    assert summary.skipped_existing == 1
    assert summary.downloaded == 1
    assert [accession for accession, _ in summary.failed] == ["0001-24-000003"]
    assert {record.accession for record in summary.succeeded} == {
        "0001-24-000001",
        "0001-24-000002",
    }
    assert raw_document_path(tmp_path, fresh).read_bytes() == b"<html>filing</html>"
    assert existing.source_url not in client.fetched
