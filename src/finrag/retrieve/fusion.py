"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).

RRF scores each document as sum(1 / (k + rank)) over the rankings it appears in. It only
uses ranks, so BM25 scores (unbounded) and cosine similarities (bounded) never have to be
put on the same scale, which is the usual pitfall when fusing scores linearly.
"""

from collections.abc import Sequence

from finrag.retrieve.models import SearchHit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], k: int = DEFAULT_RRF_K
) -> list[SearchHit]:
    """Fuse ranked hit lists; merged hits keep every retriever's rank for debugging."""
    fused: dict[str, SearchHit] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            contribution = 1.0 / (k + position)
            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = SearchHit(
                    chunk_id=hit.chunk_id,
                    score=0.0,
                    document=hit.document,
                    ranks=dict(hit.ranks),
                    scores=dict(hit.scores),
                )
            merged = fused[hit.chunk_id]
            merged.score += contribution
            merged.ranks.update(hit.ranks)
            merged.scores.update(hit.scores)
    ordered = sorted(fused.values(), key=lambda hit: hit.score, reverse=True)
    for rank, hit in enumerate(ordered, start=1):
        hit.ranks["rrf"] = rank
        hit.scores["rrf"] = hit.score
    return ordered
