"""Search the index from the command line and print ranked passages.

Example:
    uv run python scripts/search.py "What drove Azure growth?" --ticker MSFT --fiscal-year 2024
"""

import argparse
import textwrap

from finrag.config import load_settings
from finrag.retrieve.factory import build_retriever
from finrag.retrieve.models import SearchFilters, SearchHit

METHODS = ("bm25", "vector", "hybrid", "rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--method", choices=METHODS, default="rerank")
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--form", action="append", default=[])
    parser.add_argument("--fiscal-year", type=int, action="append", default=[])
    parser.add_argument("--fiscal-period", action="append", default=[])
    parser.add_argument("-k", type=int, default=5)
    return parser.parse_args()


def print_hit(position: int, hit: SearchHit) -> None:
    document = hit.document
    ranks = " ".join(f"{name}#{rank}" for name, rank in hit.ranks.items())
    print(
        f"\n[{position}] {document['ticker']} {document['form']} FY{document['fiscal_year']} "
        f"{document['fiscal_period']} | {document['section_title']} | {document['chunk_type']} "
        f"| score={hit.score:.3f} ({ranks})"
    )
    print(
        textwrap.indent(textwrap.shorten(document["text"], width=600, placeholder=" ..."), "    ")
    )
    print(f"    {document['source_url']}")


def main() -> int:
    args = parse_args()
    settings = load_settings()
    retriever = build_retriever(settings)
    filters = SearchFilters(
        tickers=tuple(ticker.upper() for ticker in args.ticker),
        forms=tuple(args.form),
        fiscal_years=tuple(args.fiscal_year),
        fiscal_periods=tuple(args.fiscal_period),
    )
    search = {
        "bm25": retriever.bm25,
        "vector": retriever.vector,
        "hybrid": retriever.hybrid,
        "rerank": retriever.hybrid_reranked,
    }[args.method]
    for position, hit in enumerate(search(args.query, filters, args.k), start=1):
        print_hit(position, hit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
