"""Cross-encoder re-ranking of retrieved candidates.

A bi-encoder embeds the query and each passage separately. A cross-encoder reads them
together, so it can compare terms directly, which is more accurate but far too slow to run
over the whole corpus. That's why it only re-orders the top few dozen fused candidates.
"""

from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from finrag.config import DEFAULT_RERANKER_MODEL
from finrag.retrieve.models import SearchHit


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        """Score (query, passage) pairs and return the best `top_n` hits, best first."""
        if not hits:
            return []
        pairs = [(query, hit.document["search_text"]) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored = []
        for hit, score in zip(hits, scores, strict=True):
            rescored.append(
                SearchHit(
                    chunk_id=hit.chunk_id,
                    score=float(score),
                    document=hit.document,
                    ranks=dict(hit.ranks),
                    scores={**hit.scores, "rerank": float(score)},
                )
            )
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        for rank, hit in enumerate(rescored, start=1):
            hit.ranks["rerank"] = rank
        return rescored[:top_n]
