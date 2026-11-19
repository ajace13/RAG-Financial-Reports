"""Bulk-index chunks (with their embeddings) into Elasticsearch."""

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

logger = logging.getLogger(__name__)


def iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    """Stream chunk dicts from a JSONL file."""
    with path.open(encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if line.strip():
                yield json.loads(line)


def append_chunks(path: Path, chunks: Iterable[dict[str, Any]]) -> int:
    """Append chunk dicts to the corpus JSONL (incremental ingestion); returns lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as chunks_file:
        for chunk in chunks:
            chunks_file.write(json.dumps(chunk) + "\n")
            written += 1
    return written


def bulk_actions(index: str, documents: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for document in documents:
        yield {"_index": index, "_id": document["chunk_id"], "_source": document}


def bulk_index(
    client: Elasticsearch, index: str, documents: Iterable[dict[str, Any]], batch_size: int = 500
) -> tuple[int, int]:
    """Index documents; returns (indexed, failed). Failures are logged, not raised."""
    indexed = failed = 0
    for ok, item in streaming_bulk(
        client,
        bulk_actions(index, documents),
        chunk_size=batch_size,
        raise_on_error=False,
        max_retries=3,
    ):
        if ok:
            indexed += 1
        else:
            failed += 1
            if failed <= 5:
                logger.error("Indexing failure: %s", item)
    return indexed, failed
