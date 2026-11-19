"""Ask a question about the indexed filings and print a cited answer.

Without ANTHROPIC_API_KEY or OPENAI_API_KEY (or with --provider none) it prints the
retrieved passages only.

Example:
    uv run python scripts/ask.py "What was 3M's capital expenditure in FY2018?" --ticker MMM
"""

import argparse
import textwrap

from finrag.config import load_settings
from finrag.generate.factory import PROVIDERS, build_generator
from finrag.generate.models import Answer, GenerationError, Passage
from finrag.generate.pipeline import answer_question
from finrag.retrieve.factory import build_retriever
from finrag.retrieve.models import SearchFilters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--provider", choices=PROVIDERS, default="auto")
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--form", action="append", default=[])
    parser.add_argument("--fiscal-year", type=int, action="append", default=[])
    parser.add_argument("--fiscal-period", action="append", default=[])
    parser.add_argument("--passages", type=int, default=8, help="passages given to the model")
    return parser.parse_args()


def print_answer(answer: Answer, passages: list[Passage]) -> None:
    """Answer text with [n] markers, then the numbered sources it cites."""
    numbering = {
        passage_id: number for number, passage_id in enumerate(answer.cited_passage_ids, 1)
    }
    rendered = []
    for segment in answer.segments:
        markers = "".join(
            f"[{numbering[citation.passage_id]}{'' if citation.verified else '?'}]"
            for citation in segment.citations
        )
        rendered.append(segment.text + markers)
    print("\n" + textwrap.fill("".join(rendered), width=100, replace_whitespace=False))
    passages_by_id = {passage.passage_id: passage for passage in passages}
    print("\nSources:")
    for passage_id, number in numbering.items():
        passage = passages_by_id.get(passage_id)
        label = passage.label if passage else passage_id
        url = passage.source_url if passage else ""
        quotes = {c.cited_text for c in answer.citations if c.passage_id == passage_id}
        print(f"  [{number}] {label}\n      {url}")
        for quote in list(quotes)[:2]:
            print(f"      > {textwrap.shorten(quote, width=140, placeholder=' ...')}")
    for note in answer.notes:
        print(f"\nNote: {note}")
    grounded = "grounded" if answer.is_grounded else "NOT grounded (no verified citations)"
    print(f"\n[{answer.provider} / {answer.model} | {grounded} | tokens {answer.usage}]")


def print_passages(passages: list[Passage]) -> None:
    print("\nRetrieval-only mode (no LLM key configured). Top passages:")
    for number, passage in enumerate(passages, 1):
        print(f"\n[{number}] {passage.label}\n    {passage.source_url}")
        print(
            textwrap.indent(textwrap.shorten(passage.text, width=500, placeholder=" ..."), "    ")
        )


def main() -> int:
    args = parse_args()
    settings = load_settings()
    generator = build_generator(settings, args.provider)
    retriever = build_retriever(settings)
    filters = SearchFilters(
        tickers=tuple(ticker.upper() for ticker in args.ticker),
        forms=tuple(args.form),
        fiscal_years=tuple(args.fiscal_year),
        fiscal_periods=tuple(args.fiscal_period),
    )
    try:
        result = answer_question(retriever, generator, args.question, filters, args.passages)
    except GenerationError as error:
        print(f"Generation failed: {error}")
        return 1
    if not result.passages:
        print("No passages matched the question and filters.")
        return 1
    if result.answer is None:
        print_passages(result.passages)
    else:
        print_answer(result.answer, result.passages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
