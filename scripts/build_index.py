"""(Re)create the Elasticsearch index and load all chunks with their embeddings."""

import argparse
import logging

from finrag.config import load_settings
from finrag.embedding import load_embedding_cache
from finrag.index.client import make_client, wait_for_cluster
from finrag.index.indexer import bulk_index, iter_chunks
from finrag.index.schema import create_index, delete_index_if_exists
from finrag.logging_utils import configure_logging

logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate", action="store_true", help="delete the existing index first (local data only)"
    )
    return parser.parse_args()


def main() -> int:
    """Index chunks; refuses to run if any chunk lacks an embedding. Returns an exit code."""
    configure_logging()
    args = parse_args()
    settings = load_settings()
    processed_dir = settings.data_dir / "processed"
    embeddings = load_embedding_cache(processed_dir / "embeddings", settings.embedding_model)
    if not embeddings:
        logger.error(
            "No embeddings for %s; run scripts/embed_chunks.py first", settings.embedding_model
        )
        return 1
    dims = len(next(iter(embeddings.values())))

    chunks = list(iter_chunks(processed_dir / "chunks.jsonl"))
    without_vectors = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in embeddings]
    if without_vectors:
        logger.error(
            "%d chunks have no embedding; run scripts/embed_chunks.py", len(without_vectors)
        )
        return 1

    client = make_client(settings)
    wait_for_cluster(client, timeout_seconds=60)
    if client.indices.exists(index=settings.es_index) and not args.recreate:
        logger.error("Index %s exists; pass --recreate to rebuild it", settings.es_index)
        return 1
    if delete_index_if_exists(client, settings.es_index):
        logger.info("Deleted existing index %s", settings.es_index)
    create_index(client, settings.es_index, dims)

    documents = ({**chunk, "embedding": embeddings[chunk["chunk_id"]].tolist()} for chunk in chunks)
    indexed, failed = bulk_index(client, settings.es_index, documents)
    client.indices.refresh(index=settings.es_index)
    count = client.count(index=settings.es_index)["count"]
    logger.info(
        "Indexed %d chunks (%d failed); index %s now holds %d docs",
        indexed,
        failed,
        settings.es_index,
        count,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
