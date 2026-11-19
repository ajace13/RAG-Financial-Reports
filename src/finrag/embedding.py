"""Sentence-Transformers embedder plus an on-disk embedding cache keyed by chunk_id."""

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from finrag.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# BGE v1.5 retrieval works best with an instruction prefix on queries (not on passages).
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    """Encodes passages and queries into L2-normalized vectors (cosine = dot product)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")

    @property
    def dims(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([self.query_prefix + query])[0]

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Token counts per text, used to report how many passages get truncated."""
        encoded = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()


def embedding_cache_paths(cache_dir: Path, model_name: str) -> tuple[Path, Path]:
    slug = model_slug(model_name)
    return cache_dir / f"{slug}.npy", cache_dir / f"{slug}.ids.json"


def load_embedding_cache(cache_dir: Path, model_name: str) -> dict[str, np.ndarray]:
    """Load cached vectors for a model as {chunk_id: vector}; empty if no cache yet."""
    vectors_path, ids_path = embedding_cache_paths(cache_dir, model_name)
    if not vectors_path.exists() or not ids_path.exists():
        return {}
    vectors = np.load(vectors_path)
    chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(chunk_ids) != len(vectors):
        raise ValueError(f"Embedding cache is inconsistent: {len(chunk_ids)} ids vs {len(vectors)}")
    return dict(zip(chunk_ids, vectors, strict=True))


def save_embedding_cache(
    cache_dir: Path, model_name: str, embeddings: dict[str, np.ndarray]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors_path, ids_path = embedding_cache_paths(cache_dir, model_name)
    chunk_ids = list(embeddings)
    matrix = np.stack([embeddings[chunk_id] for chunk_id in chunk_ids]).astype(np.float32)
    np.save(vectors_path, matrix)
    ids_path.write_text(json.dumps(chunk_ids), encoding="utf-8")
