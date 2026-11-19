"""Retrieval ablation on FinanceBench: BM25 vs vector vs hybrid vs hybrid + rerank.

Two settings:
- document: search restricted to the one filing the question is about ("given this 10-K").
- company: search across all of that company's filings in the index, which also tests
  whether retrieval lands on the right fiscal year.

Relevance labels come from FinanceBench evidence text (see finrag.evaluation.relevance).
Questions whose evidence matches no chunk are reported as unjudgeable and excluded, not
silently dropped. Writes eval/results/retrieval_ablation.{json,md}.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from finrag.config import PROJECT_ROOT, load_settings
from finrag.embedding import Embedder
from finrag.evaluation.financebench import load_questions
from finrag.evaluation.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from finrag.evaluation.relevance import relevant_chunk_ids
from finrag.index.client import make_client
from finrag.index.indexer import iter_chunks
from finrag.logging_utils import configure_logging
from finrag.retrieve.models import SearchFilters
from finrag.retrieve.rerank import Reranker
from finrag.retrieve.retriever import Retriever

logger = logging.getLogger("run_retrieval_eval")

K_VALUES = (1, 5, 10)
METHODS = ("bm25", "vector", "hybrid", "hybrid+rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", default=["document", "company"])
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions"
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--tag", default="", help="suffix for result files, e.g. 'bge-reranker' for a variant run"
    )
    return parser.parse_args()


def build_qrels(questions: list, doc_map: dict[str, str], chunks_path: Path) -> dict[str, set[str]]:
    """Relevant chunk ids per question, computed within the question's own filing."""
    needed = {doc_map[question.doc.doc_name] for question in questions}
    chunks_by_accession: dict[str, list[dict]] = defaultdict(list)
    for chunk in iter_chunks(chunks_path):
        if chunk["accession"] in needed:
            chunks_by_accession[chunk["accession"]].append(chunk)
    return {
        question.question_id: relevant_chunk_ids(
            chunks_by_accession[doc_map[question.doc.doc_name]], question.evidence_texts
        )
        for question in questions
    }


def summarize(per_question: list[dict]) -> dict[str, float]:
    summary = {f"hit@{k}": mean(row[f"hit@{k}"] for row in per_question) for k in K_VALUES}
    summary["mrr@10"] = mean(row["mrr@10"] for row in per_question)
    summary["ndcg@10"] = mean(row["ndcg@10"] for row in per_question)
    summary["latency_ms"] = mean(row["latency_ms"] for row in per_question)
    return summary


def render_markdown(results: dict, judged: int, total: int, config: dict) -> str:
    lines = [
        "# Retrieval ablation: FinanceBench open sample",
        "",
        f"Judgeable questions: **{judged} / {total}** (10-K/10-Q questions whose evidence "
        "text matched at least one chunk of the cited filing).",
        "",
        f"Embedding model: `{config['embedding_model']}` | "
        f"Reranker: `{config['reranker_model']}` | "
        f"candidates per retriever: {config['candidates']} | "
        f"rerank pool: {config['rerank_pool']}",
        "",
    ]
    for setting, by_method in results.items():
        lines += [
            f"## Setting: {setting}",
            "",
            "| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | Latency (ms) |",
            "|---|---|---|---|---|---|---|",
        ]
        for method, summary in by_method.items():
            lines.append(
                f"| {method} | {summary['hit@1']:.3f} | {summary['hit@5']:.3f} | "
                f"{summary['hit@10']:.3f} | {summary['mrr@10']:.3f} | {summary['ndcg@10']:.3f} | "
                f"{summary['latency_ms']:.0f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Run every retriever over every judgeable question in each setting and save results."""
    configure_logging()
    args = parse_args()
    settings = load_settings()
    eval_dir = settings.data_dir / "eval"
    doc_map = json.loads((eval_dir / "financebench_doc_map.json").read_text(encoding="utf-8"))
    all_questions = load_questions(eval_dir / "financebench_open_source.jsonl")
    questions = [question for question in all_questions if question.doc.doc_name in doc_map]
    logger.info(
        "%d of %d questions refer to a resolved 10-K/10-Q (others cite 8-Ks / earnings releases)",
        len(questions),
        len(all_questions),
    )
    if args.limit:
        questions = questions[: args.limit]

    qrels = build_qrels(questions, doc_map, settings.data_dir / "processed" / "chunks.jsonl")
    judgeable = [question for question in questions if qrels[question.question_id]]
    unjudgeable = [
        question.question_id for question in questions if not qrels[question.question_id]
    ]
    logger.info(
        "Judgeable: %d; unjudgeable (evidence matched no chunk): %d %s",
        len(judgeable),
        len(unjudgeable),
        unjudgeable,
    )

    retriever = Retriever(
        make_client(settings),
        settings.es_index,
        Embedder(settings.embedding_model),
        Reranker(settings.reranker_model),
    )
    methods = {
        "bm25": retriever.bm25,
        "vector": retriever.vector,
        "hybrid": retriever.hybrid,
        "hybrid+rerank": retriever.hybrid_reranked,
    }
    results: dict[str, dict[str, dict]] = {}
    per_question_rows: list[dict] = []
    for setting in args.settings:
        results[setting] = {}
        for method_name in args.methods:
            rows = []
            for question in judgeable:
                accession = doc_map[question.doc.doc_name]
                filters = (
                    SearchFilters(accessions=(accession,))
                    if setting == "document"
                    else SearchFilters(tickers=(question.doc.ticker,))
                )
                started = time.perf_counter()
                hits = methods[method_name](question.question, filters, max(K_VALUES))
                latency_ms = 1000 * (time.perf_counter() - started)
                ranked_ids = [hit.chunk_id for hit in hits]
                relevant = qrels[question.question_id]
                row = {
                    "setting": setting,
                    "method": method_name,
                    "question_id": question.question_id,
                    "doc_name": question.doc.doc_name,
                    "latency_ms": latency_ms,
                    "mrr@10": reciprocal_rank(ranked_ids, relevant, 10),
                    "ndcg@10": ndcg_at_k(ranked_ids, relevant, 10),
                    "first_relevant_rank": next(
                        (
                            rank
                            for rank, chunk_id in enumerate(ranked_ids, 1)
                            if chunk_id in relevant
                        ),
                        None,
                    ),
                    "top_accessions": [hit.document["accession"] for hit in hits[:5]],
                }
                row.update({f"hit@{k}": hit_at_k(ranked_ids, relevant, k) for k in K_VALUES})
                rows.append(row)
            results[setting][method_name] = summarize(rows)
            per_question_rows.extend(rows)
            logger.info(
                "%s / %s: %s",
                setting,
                method_name,
                {key: round(value, 3) for key, value in results[setting][method_name].items()},
            )

    config = {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "candidates": retriever.config.candidates_per_retriever,
        "rerank_pool": retriever.config.rerank_candidates,
    }
    output_dir = PROJECT_ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    (output_dir / f"retrieval_ablation{suffix}.json").write_text(
        json.dumps(
            {
                "config": config,
                "judgeable": len(judgeable),
                "evaluated": len(questions),
                "unjudgeable_question_ids": unjudgeable,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / f"retrieval_per_question{suffix}.jsonl").write_text(
        "\n".join(json.dumps(row) for row in per_question_rows) + "\n", encoding="utf-8"
    )
    (output_dir / f"retrieval_ablation{suffix}.md").write_text(
        render_markdown(results, len(judgeable), len(questions), config), encoding="utf-8"
    )
    logger.info("Wrote %s", output_dir / f"retrieval_ablation{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
