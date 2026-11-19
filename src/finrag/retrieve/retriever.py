"""Composes the retrieval stages. Each stage is its own method so they can be compared."""

from dataclasses import dataclass

from elasticsearch import Elasticsearch

from finrag.embedding import Embedder
from finrag.retrieve.dense import knn_search
from finrag.retrieve.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from finrag.retrieve.lexical import bm25_search
from finrag.retrieve.models import SearchFilters, SearchHit
from finrag.retrieve.rerank import Reranker


@dataclass(frozen=True)
class RetrievalConfig:
    candidates_per_retriever: int = 50
    rerank_candidates: int = 30
    final_k: int = 10
    rrf_k: int = DEFAULT_RRF_K


class Retriever:
    def __init__(
        self,
        client: Elasticsearch,
        index: str,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.client = client
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def bm25(self, query: str, filters: SearchFilters, size: int | None = None) -> list[SearchHit]:
        return bm25_search(self.client, self.index, query, filters, size or self.config.final_k)

    def vector(
        self, query: str, filters: SearchFilters, size: int | None = None
    ) -> list[SearchHit]:
        query_vector = self.embedder.embed_query(query)
        return knn_search(
            self.client, self.index, query_vector, filters, size or self.config.final_k
        )

    def hybrid(
        self, query: str, filters: SearchFilters, size: int | None = None
    ) -> list[SearchHit]:
        """RRF over the top candidates of BM25 and kNN."""
        candidates = self.config.candidates_per_retriever
        fused = reciprocal_rank_fusion(
            [self.bm25(query, filters, candidates), self.vector(query, filters, candidates)],
            k=self.config.rrf_k,
        )
        return fused[: size or self.config.final_k]

    def hybrid_reranked(
        self, query: str, filters: SearchFilters, size: int | None = None
    ) -> list[SearchHit]:
        """Hybrid candidates re-ordered by the cross-encoder."""
        if self.reranker is None:
            raise ValueError("hybrid_reranked requires a Reranker")
        candidates = self.hybrid(query, filters, self.config.rerank_candidates)
        return self.reranker.rerank(query, candidates, size or self.config.final_k)
