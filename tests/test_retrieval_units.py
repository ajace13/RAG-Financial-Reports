from finrag.retrieve.fusion import reciprocal_rank_fusion
from finrag.retrieve.models import SearchFilters, SearchHit


def make_hits(retriever: str, chunk_ids: list[str]) -> list[SearchHit]:
    return [
        SearchHit(chunk_id=chunk_id, score=1.0, document={"id": chunk_id}, ranks={retriever: rank})
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def test_rrf_rewards_agreement_and_records_ranks() -> None:
    bm25 = make_hits("bm25", ["a", "b", "c"])
    vector = make_hits("vector", ["c", "a", "d"])
    fused = reciprocal_rank_fusion([bm25, vector], k=60)
    assert [hit.chunk_id for hit in fused][:2] == ["a", "c"]
    top = fused[0]
    assert top.ranks == {"bm25": 1, "vector": 2, "rrf": 1}
    assert top.score == 1 / 61 + 1 / 62
    assert {hit.chunk_id for hit in fused} == {"a", "b", "c", "d"}


def test_rrf_does_not_mutate_inputs() -> None:
    bm25 = make_hits("bm25", ["a"])
    reciprocal_rank_fusion([bm25, make_hits("vector", ["a"])])
    assert bm25[0].ranks == {"bm25": 1}


def test_filters_to_es_clauses() -> None:
    filters = SearchFilters(tickers=("MSFT",), fiscal_years=(2024, 2023))
    assert filters.to_es_filters() == [
        {"terms": {"ticker": ["MSFT"]}},
        {"terms": {"fiscal_year": [2024, 2023]}},
    ]
    assert SearchFilters().to_es_filters() == []
