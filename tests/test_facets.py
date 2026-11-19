from typing import Any

from finrag.index.facets import fetch_facets
from finrag.retrieve.models import SearchFilters


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        return {
            "aggregations": {
                "tickers": {
                    "buckets": [
                        {"key": "MSFT", "company": {"buckets": [{"key": "MICROSOFT CORP"}]}},
                        {"key": "AMZN", "company": {"buckets": [{"key": "AMAZON COM INC"}]}},
                    ]
                },
                "forms": {"buckets": [{"key": "10-Q"}, {"key": "10-K"}]},
                "fiscal_years": {"buckets": [{"key": 2023}, {"key": 2025}, {"key": 2024}]},
            }
        }


def test_fetch_facets_sorts_and_scopes() -> None:
    client = FakeClient()
    facets = fetch_facets(client, "index", SearchFilters(tickers=("MSFT",)))
    assert facets.companies == {"AMZN": "AMAZON COM INC", "MSFT": "MICROSOFT CORP"}
    assert facets.forms == ["10-K", "10-Q"]
    assert facets.fiscal_years == [2025, 2024, 2023]
    assert client.calls[0]["query"] == {"bool": {"filter": [{"terms": {"ticker": ["MSFT"]}}]}}
    assert client.calls[0]["size"] == 0
