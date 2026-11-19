"""Construct the query-time Retriever from settings (shared by the app, CLIs and eval)."""

from finrag.config import Settings
from finrag.embedding import Embedder
from finrag.index.client import make_client
from finrag.retrieve.rerank import Reranker
from finrag.retrieve.retriever import Retriever


def build_retriever(settings: Settings) -> Retriever:
    """Retriever with the query embedder and re-ranker on `settings.retrieval_device`.

    That setting only covers question-time models. Bulk corpus embedding
    (scripts/embed_chunks.py) always uses the fastest available device.
    """
    return Retriever(
        make_client(settings),
        settings.es_index,
        Embedder(settings.embedding_model, device=settings.retrieval_device),
        Reranker(settings.reranker_model, device=settings.retrieval_device),
    )
