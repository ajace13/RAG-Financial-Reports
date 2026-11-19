"""Embed every chunk's search_text, reusing cached vectors from earlier runs.

Writes data/processed/embeddings/<model-slug>.npy and .ids.json.
"""

import logging
import time

import numpy as np

from finrag.config import load_settings
from finrag.embedding import Embedder, load_embedding_cache, save_embedding_cache
from finrag.index.indexer import iter_chunks
from finrag.logging_utils import configure_logging

logger = logging.getLogger("embed_chunks")

ENCODE_BATCH = 2048  # chunks per encode call; the cache is saved after every batch


def main() -> int:
    """Embed missing chunks and report how many exceed the model's token window."""
    configure_logging()
    settings = load_settings()
    processed_dir = settings.data_dir / "processed"
    cache_dir = processed_dir / "embeddings"

    chunks = {
        chunk["chunk_id"]: chunk["search_text"]
        for chunk in iter_chunks(processed_dir / "chunks.jsonl")
    }
    cache = load_embedding_cache(cache_dir, settings.embedding_model)
    stale = set(cache) - set(chunks)
    for chunk_id in stale:
        del cache[chunk_id]
    missing = [chunk_id for chunk_id in chunks if chunk_id not in cache]
    logger.info(
        "%d chunks: %d cached, %d to embed, %d stale cache entries removed",
        len(chunks),
        len(chunks) - len(missing),
        len(missing),
        len(stale),
    )

    embedder = Embedder(settings.embedding_model)
    logger.info(
        "Model %s on %s (dims=%d, max_tokens=%d)",
        embedder.model_name,
        embedder.model.device,
        embedder.dims,
        embedder.max_tokens,
    )

    sample = list(chunks.values())[:: max(1, len(chunks) // 2000)]
    token_counts = np.array(embedder.count_tokens(sample))
    logger.info(
        "Token length on a %d-chunk sample: p50=%d p95=%d max=%d; %.1f%% exceed %d (truncated)",
        len(sample),
        np.percentile(token_counts, 50),
        np.percentile(token_counts, 95),
        token_counts.max(),
        100 * np.mean(token_counts > embedder.max_tokens),
        embedder.max_tokens,
    )

    started = time.time()
    for start in range(0, len(missing), ENCODE_BATCH):
        batch_ids = missing[start : start + ENCODE_BATCH]
        vectors = embedder.embed_documents([chunks[chunk_id] for chunk_id in batch_ids])
        cache.update(zip(batch_ids, vectors, strict=True))
        save_embedding_cache(cache_dir, settings.embedding_model, cache)
        done = start + len(batch_ids)
        rate = done / max(time.time() - started, 1e-6)
        logger.info("Embedded %d/%d (%.0f chunks/s)", done, len(missing), rate)
    if not missing:
        save_embedding_cache(cache_dir, settings.embedding_model, cache)
    logger.info("Embedding cache now holds %d vectors", len(cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
