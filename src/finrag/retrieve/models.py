"""Shared retrieval data types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchFilters:
    """Metadata restrictions applied identically to BM25 and kNN queries. Empty = no filter."""

    tickers: tuple[str, ...] = ()
    forms: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    accessions: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()

    def to_es_filters(self) -> list[dict[str, Any]]:
        clauses = []
        for field_name, values in (
            ("ticker", self.tickers),
            ("form", self.forms),
            ("fiscal_year", self.fiscal_years),
            ("fiscal_period", self.fiscal_periods),
            ("accession", self.accessions),
            ("chunk_type", self.chunk_types),
        ):
            if values:
                clauses.append({"terms": {field_name: list(values)}})
        return clauses


@dataclass
class SearchHit:
    """One retrieved chunk. `ranks` records its 1-based rank in each retriever that found it."""

    chunk_id: str
    score: float
    document: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
