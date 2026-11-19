"""BM25 retrieval over the `search_text` field."""

from typing import Any

from elasticsearch import Elasticsearch

from finrag.retrieve.models import SearchFilters, SearchHit

EXCLUDED_SOURCE_FIELDS = ["embedding"]


def hits_from_response(response: Any, retriever_name: str) -> list[SearchHit]:
    """Convert an Elasticsearch response into SearchHits ranked from 1."""
    hits = []
    for rank, raw_hit in enumerate(response["hits"]["hits"], start=1):
        score = float(raw_hit["_score"] or 0.0)
        hits.append(
            SearchHit(
                chunk_id=raw_hit["_id"],
                score=score,
                document=raw_hit["_source"],
                ranks={retriever_name: rank},
                scores={retriever_name: score},
            )
        )
    return hits


def bm25_search(
    client: Elasticsearch, index: str, query: str, filters: SearchFilters, size: int
) -> list[SearchHit]:
    """Top-`size` chunks by BM25 on `search_text`, restricted by `filters`."""
    response = client.search(
        index=index,
        query={
            "bool": {
                "must": [{"match": {"search_text": {"query": query}}}],
                "filter": filters.to_es_filters(),
            }
        },
        size=size,
        source_excludes=EXCLUDED_SOURCE_FIELDS,
    )
    return hits_from_response(response, "bm25")
