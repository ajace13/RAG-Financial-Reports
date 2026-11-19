"""Question -> retrieved passages -> grounded answer (or passages only, without a generator)."""

from dataclasses import dataclass

from finrag.generate.models import Answer, Generator, Passage
from finrag.generate.passages import passages_from_hits
from finrag.retrieve.models import SearchFilters, SearchHit
from finrag.retrieve.retriever import Retriever

DEFAULT_PASSAGE_COUNT = 8


@dataclass
class QAResult:
    question: str
    hits: list[SearchHit]
    passages: list[Passage]
    answer: Answer | None  # None in retrieval-only mode


def answer_question(
    retriever: Retriever,
    generator: Generator | None,
    question: str,
    filters: SearchFilters,
    passage_count: int = DEFAULT_PASSAGE_COUNT,
) -> QAResult:
    """Retrieve with hybrid search + re-ranking, then generate if a generator is configured."""
    hits = retriever.hybrid_reranked(question, filters, passage_count)
    passages = passages_from_hits(hits)
    answer = generator.generate(question, passages) if generator is not None and passages else None
    return QAResult(question=question, hits=hits, passages=passages, answer=answer)
