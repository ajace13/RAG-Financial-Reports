"""Elasticsearch index definition for filing chunks."""

from typing import Any

from elasticsearch import Elasticsearch

KEYWORD = {"type": "keyword"}


def index_settings() -> dict[str, Any]:
    """Single shard, no replicas: one local node, and it keeps cluster health green."""
    return {"number_of_shards": 1, "number_of_replicas": 0}


def index_mappings(embedding_dims: int) -> dict[str, Any]:
    """Field mappings. `search_text` carries BM25; `embedding` carries kNN.

    `dynamic: strict` makes a chunk with an unexpected field fail loudly, so it can't
    silently create a field nobody queries. Vectors use plain float HNSW (not the quantized
    default) so the retrieval evaluation measures the embedding model, not quantization.
    """
    return {
        "dynamic": "strict",
        "properties": {
            "chunk_id": KEYWORD,
            "accession": KEYWORD,
            "ticker": KEYWORD,
            "cik": KEYWORD,
            "company_name": KEYWORD,
            "form": KEYWORD,
            "is_amendment": {"type": "boolean"},
            "fiscal_year": {"type": "integer"},
            "fiscal_period": KEYWORD,
            "fiscal_year_end": KEYWORD,
            "period_of_report": {"type": "date"},
            "filing_date": {"type": "date"},
            "part": KEYWORD,
            "item": KEYWORD,
            "section_title": {"type": "text", "fields": {"raw": KEYWORD}},
            "chunk_type": KEYWORD,
            "units": KEYWORD,
            "seq": {"type": "integer"},
            "text": {"type": "text", "index": False},
            "search_text": {"type": "text", "analyzer": "english"},
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dims,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw"},
            },
            "source_url": {"type": "keyword", "index": False},
        },
    }


def delete_index_if_exists(client: Elasticsearch, index: str) -> bool:
    """Delete `index` if present; returns whether anything was deleted."""
    if not client.indices.exists(index=index):
        return False
    client.indices.delete(index=index)
    return True


def create_index(client: Elasticsearch, index: str, embedding_dims: int) -> None:
    client.indices.create(
        index=index, settings=index_settings(), mappings=index_mappings(embedding_dims)
    )
