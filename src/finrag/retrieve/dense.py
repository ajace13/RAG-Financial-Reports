"""Dense kNN retrieval over the `embedding` field."""

import numpy as np
from elasticsearch import Elasticsearch

from finrag.retrieve.lexical import EXCLUDED_SOURCE_FIELDS, hits_from_response
from finrag.retrieve.models import SearchFilters, SearchHit

# HNSW explores this many candidates per shard before returning the top k. A generous value
# matters with restrictive filters (e.g. one filing of ~400 chunks out of ~47k).
MIN_NUM_CANDIDATES = 500


def knn_search(
    client: Elasticsearch,
    index: str,
    query_vector: np.ndarray,
    filters: SearchFilters,
    size: int,
) -> list[SearchHit]:
    """Top-`size` chunks by cosine similarity, with filters applied inside the HNSW search."""
    response = client.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": query_vector.tolist(),
            "k": size,
            "num_candidates": max(MIN_NUM_CANDIDATES, size * 10),
            "filter": filters.to_es_filters(),
        },
        size=size,
        source_excludes=EXCLUDED_SOURCE_FIELDS,
    )
    return hits_from_response(response, "vector")
