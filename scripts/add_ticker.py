"""Fetch, parse, embed and index companies end to end (incremental; skips indexed filings).

Examples:
    uv run python scripts/add_ticker.py NVDA
    uv run python scripts/add_ticker.py MSFT --full-history     # extend to full EDGAR history
    uv run python scripts/add_ticker.py NVDA AAPL GOOGL         # several, one after another
"""

import argparse
import logging

from finrag.config import load_settings
from finrag.ingest.service import IngestError, IngestProgress, IngestRequest, ingest_company
from finrag.logging_utils import configure_logging

logger = logging.getLogger("add_ticker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tickers", nargs="+")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="all HTML-era filings (~2001 onward) instead of the last 5 years",
    )
    return parser.parse_args()


def log_progress(progress: IngestProgress) -> None:
    counter = f" {progress.done}/{progress.total}" if progress.total else ""
    logger.info("%s%s %s", progress.stage, counter, progress.detail)


def main() -> int:
    """Ingest each ticker; returns 1 if any ticker failed entirely."""
    configure_logging()
    args = parse_args()
    settings = load_settings()
    failures = 0
    for ticker in args.tickers:
        request = IngestRequest(ticker=ticker, full_history=args.full_history)
        try:
            result = ingest_company(settings, request, log_progress)
        except IngestError as error:
            logger.error("%s: %s", ticker, error)
            failures += 1
            continue
        logger.info(
            "%s (%s, %s): %d filings in scope, %d already indexed, %d added (%d chunks), "
            "%d non-HTML skipped, %d failed",
            result.ticker,
            result.company_name,
            request.scope_label,
            result.filings_found,
            result.already_indexed,
            result.filings_added,
            result.chunks_added,
            result.skipped_non_html,
            len(result.failed),
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
