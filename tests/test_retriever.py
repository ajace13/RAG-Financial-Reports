"""Retriever composition, with BM25/kNN/model calls replaced by fakes."""

from typing import Any

import numpy as np
import pytest

from finrag.retrieve import retriever as retriever_module
from finrag.retrieve.models import SearchFilters, SearchHit
from finrag.retrieve.retriever import RetrievalConfig, Retriever


def make_hits(name: str, chunk_ids: list[str]) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=chunk_id, score=1.0, document={"search_text": chunk_id}, ranks={name: rank}
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


class FakeReranker:
    """Prefers chunk ids that sort last alphabetically, to prove re-ordering happens."""

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return sorted(hits, key=lambda hit: hit.chunk_id, reverse=True)[:top_n]


@pytest.fixture
def patched_search(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_bm25(client, index, query, filters, size):
        calls["bm25_size"] = size
        return make_hits("bm25", ["a", "b", "c"])[:size]

    def fake_knn(client, index, query_vector, filters, size):
        calls["knn_size"] = size
        return make_hits("vector", ["c", "d", "a"])[:size]

    monkeypatch.setattr(retriever_module, "bm25_search", fake_bm25)
    monkeypatch.setattr(retriever_module, "knn_search", fake_knn)
    return calls


def test_hybrid_fuses_candidate_pools(patched_search: dict[str, Any]) -> None:
    config = RetrievalConfig(candidates_per_retriever=3, final_k=2)
    retriever = Retriever(None, "index", FakeEmbedder(), config=config)
    hits = retriever.hybrid("query", SearchFilters())
    assert [hit.chunk_id for hit in hits] == ["a", "c"]
    assert patched_search == {"bm25_size": 3, "knn_size": 3}


def test_hybrid_reranked_reorders_pool(patched_search: dict[str, Any]) -> None:
    config = RetrievalConfig(candidates_per_retriever=3, rerank_candidates=4, final_k=2)
    retriever = Retriever(None, "index", FakeEmbedder(), FakeReranker(), config)
    assert [hit.chunk_id for hit in retriever.hybrid_reranked("query", SearchFilters())] == [
        "d",
        "c",
    ]


def test_hybrid_reranked_requires_reranker(patched_search: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Retriever(None, "index", FakeEmbedder()).hybrid_reranked("query", SearchFilters())
