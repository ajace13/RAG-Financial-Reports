"""Live checks against the local index (run with `pytest -m integration` after build_index)."""

import pytest

from finrag.config import load_settings
from finrag.index.client import make_client
from finrag.retrieve.lexical import bm25_search
from finrag.retrieve.models import SearchFilters


@pytest.mark.integration
def test_bm25_respects_filters() -> None:
    settings = load_settings()
    client = make_client(settings)
    filters = SearchFilters(tickers=("MSFT",), forms=("10-K",))
    hits = bm25_search(client, settings.es_index, "Azure revenue growth", filters, size=5)
    assert hits, "expected MSFT 10-K passages about Azure"
    assert all(hit.document["ticker"] == "MSFT" for hit in hits)
    assert all(hit.document["form"] == "10-K" for hit in hits)
    assert "embedding" not in hits[0].document
