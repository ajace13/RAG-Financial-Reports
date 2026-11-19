"""Turn retrieved chunks into passages with fine-grained citable blocks.

Citations point at whole blocks, so block granularity sets citation precision: one block per
table row (so a citation lands on the exact line item) and one per sentence for prose.
"""

from collections.abc import Sequence
from typing import Any

from finrag.chunking import split_sentences
from finrag.generate.models import Passage
from finrag.retrieve.models import SearchHit


def passage_label(document: dict[str, Any]) -> str:
    """Human-readable source label, e.g. 'MICROSOFT CORP (MSFT) 10-K FY2024 | Item 7. ...'."""
    fiscal_year, fiscal_period = document.get("fiscal_year"), document.get("fiscal_period")
    if not fiscal_year:
        period = f"period ending {document.get('period_of_report')}"
    elif fiscal_period in (None, "FY"):
        period = f"FY{fiscal_year}"
    else:
        period = f"FY{fiscal_year} {fiscal_period}"
    section = document.get("section_title") or "Unknown section"
    item = document.get("item")
    section_label = (
        f"Item {item}. {section}" if item and item not in ("front", "unknown") else section
    )
    units = f" ({document['units']})" if document.get("units") else ""
    filing = f"{document['company_name']} ({document['ticker']}) {document['form']} {period}"
    return f"{filing} | {section_label}{units}"


def split_into_blocks(text: str, chunk_type: str) -> tuple[str, ...]:
    """Table chunks split into lines (title + rows); prose into sentences."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if chunk_type == "table":
        return tuple(lines)
    return tuple(
        sentence for line in lines for sentence in split_sentences(line) if sentence.strip()
    )


def passage_from_hit(hit: SearchHit) -> Passage:
    document = hit.document
    return Passage(
        passage_id=hit.chunk_id,
        label=passage_label(document),
        source_url=document["source_url"],
        text=document["text"],
        blocks=split_into_blocks(document["text"], document["chunk_type"]),
    )


def passages_from_hits(hits: Sequence[SearchHit]) -> list[Passage]:
    return [passage_from_hit(hit) for hit in hits]
