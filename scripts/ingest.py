"""Download recent 10-K / 10-Q filings for a list of tickers into data/raw/."""

import argparse
import logging
from datetime import date, timedelta

from finrag.config import load_settings
from finrag.ingest.edgar_client import EdgarClient
from finrag.ingest.filings import fetch_company_filings, load_ticker_to_cik, select_filings
from finrag.ingest.storage import download_filings, merge_into_manifest
from finrag.logging_utils import configure_logging

logger = logging.getLogger("ingest")

DEFAULT_TICKERS = ["MSFT", "AMZN", "JPM", "PEP", "MMM"]
DEFAULT_FORMS = ["10-K", "10-Q"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--forms", nargs="+", default=DEFAULT_FORMS)
    parser.add_argument("--years", type=float, default=3.0, help="look-back window by filing date")
    return parser.parse_args()


def main() -> int:
    """Discover, filter and download filings; update the manifest. Returns an exit code."""
    configure_logging()
    args = parse_args()
    settings = load_settings()
    client = EdgarClient(settings.sec_user_agent or "")
    raw_dir = settings.data_dir / "raw"
    filed_since = date.today() - timedelta(days=round(365.25 * args.years))

    ticker_to_cik = load_ticker_to_cik(client)
    failures = 0
    for ticker in (ticker.upper() for ticker in args.tickers):
        if ticker not in ticker_to_cik:
            logger.error("Ticker %s not found in SEC ticker map; skipping", ticker)
            failures += 1
            continue
        company, records = fetch_company_filings(client, ticker, ticker_to_cik[ticker], filed_since)
        selected = select_filings(records, args.forms, filed_since)
        logger.info("%s (%s): %d filings selected", ticker, company.name, len(selected))
        summary = download_filings(client, selected, raw_dir)
        manifest_size = merge_into_manifest(raw_dir, summary.succeeded)
        logger.info(
            "%s: downloaded %d, already present %d, failed %d; manifest has %d filings",
            ticker,
            summary.downloaded,
            summary.skipped_existing,
            len(summary.failed),
            manifest_size,
        )
        failures += len(summary.failed)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
