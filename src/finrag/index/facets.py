"""Distinct filter values (companies, forms, fiscal years) from the index, for UI filters."""

from dataclasses import dataclass, field

from elasticsearch import Elasticsearch

from finrag.retrieve.models import SearchFilters

MAX_BUCKETS = 500


@dataclass(frozen=True)
class Facets:
    companies: dict[str, str] = field(default_factory=dict)  # ticker -> company name
    forms: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)  # newest first


def fetch_facets(client: Elasticsearch, index: str, filters: SearchFilters) -> Facets:
    """Aggregate filter values; `filters` scopes them (e.g. years available for chosen tickers)."""
    response = client.search(
        index=index,
        size=0,
        query={"bool": {"filter": filters.to_es_filters()}},
        aggs={
            "tickers": {
                "terms": {"field": "ticker", "size": MAX_BUCKETS},
                "aggs": {"company": {"terms": {"field": "company_name", "size": 1}}},
            },
            "forms": {"terms": {"field": "form", "size": 20}},
            "fiscal_years": {"terms": {"field": "fiscal_year", "size": MAX_BUCKETS}},
        },
    )
    aggregations = response["aggregations"]
    companies = {
        bucket["key"]: (
            bucket["company"]["buckets"][0]["key"] if bucket["company"]["buckets"] else ""
        )
        for bucket in aggregations["tickers"]["buckets"]
    }
    return Facets(
        companies=dict(sorted(companies.items())),
        forms=sorted(bucket["key"] for bucket in aggregations["forms"]["buckets"]),
        fiscal_years=sorted(
            (int(bucket["key"]) for bucket in aggregations["fiscal_years"]["buckets"]),
            reverse=True,
        ),
    )
