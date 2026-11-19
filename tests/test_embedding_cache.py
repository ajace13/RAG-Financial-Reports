from pathlib import Path

import numpy as np

from finrag.embedding import load_embedding_cache, model_slug, save_embedding_cache


def test_model_slug() -> None:
    assert model_slug("BAAI/bge-base-en-v1.5") == "baai-bge-base-en-v1-5"


def test_cache_round_trip(tmp_path: Path) -> None:
    embeddings = {"a": np.array([1.0, 0.0], dtype=np.float32), "b": np.array([0.0, 1.0])}
    save_embedding_cache(tmp_path, "some/model", embeddings)
    loaded = load_embedding_cache(tmp_path, "some/model")
    assert set(loaded) == {"a", "b"}
    np.testing.assert_allclose(loaded["b"], [0.0, 1.0])
    assert load_embedding_cache(tmp_path, "other/model") == {}
