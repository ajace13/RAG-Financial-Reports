"""Ranking metrics with binary relevance."""

import math
from collections.abc import Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant chunk is in the top k (FinanceBench needs one evidence passage)."""
    return float(any(chunk_id in relevant for chunk_id in ranked_ids[:k]))


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """1/rank of the first relevant chunk within the top k, else 0."""
    for position, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary gains."""
    gains = [
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant
    ]
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return sum(gains) / ideal if ideal else 0.0
