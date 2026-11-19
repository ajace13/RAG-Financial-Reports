import math

import pytest

from finrag.evaluation.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from finrag.evaluation.relevance import normalize_tokens, relevance_score, relevant_chunk_ids

PDF_EVIDENCE = """Consolidated Statement of Cash Flow s
Years ended December 31
(Millions)
2018
2017
Purchases of property, plant and equipment (PP&E)
(1,577)
(1,373)
Proceeds from sale of PP&E and other assets
262
49
Acquisitions, net of cash acquired
13
(2,023)"""

HTML_CHUNK = (
    "Cash Flows from Investing Activities:\n(Millions) | 2018 | 2017\n"
    "Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)\n"
    "Proceeds from sale of PP&E and other assets | 262 | 49\n"
    "Acquisitions, net of cash acquired | 13 | (2,023)"
)


def test_normalize_tokens_strips_thousands_separators() -> None:
    assert normalize_tokens("$(1,577) Purchases") == ["1577", "purchases"]


def test_html_table_chunk_matches_pdf_evidence() -> None:
    assert relevance_score(HTML_CHUNK, PDF_EVIDENCE) >= 0.5
    unrelated = "Risk factors: our business depends on attracting and retaining talented employees."
    assert relevance_score(unrelated, PDF_EVIDENCE) == 0.0


def test_short_evidence_sentence_inside_long_chunk() -> None:
    evidence = "In 2022, 40% of our revenues were earned pursuant to U.S. government contracts"
    chunk = (
        "Risks Related to Our Contracts. We conduct a significant portion of our business "
        "pursuant to U.S. government contracts. In 2022, 40% of our revenues were earned "
        "pursuant to U.S. government contracts, which include FMS through the U.S. government."
    )
    assert relevance_score(chunk, evidence) == 1.0


def test_numbers_match_when_pdf_lost_word_spacing() -> None:
    evidence = (
        "Revenues:\nProducts\n$\n183,910 $\n180,063 $\n173,377\nPremiums\n8,184\n3,558\n3,069"
    )
    evidence += "\nServices\n1,825\n1,144\n1,080\nNetinvestmentincome\n660\n21\n20"
    chunk = (
        "Revenues: | 2018 | 2017 | 2016\nProducts | $183,910 | $180,063 | $173,377\n"
        "Premiums | 8,184 | 3,558 | 3,069\nServices | 1,825 | 1,144 | 1,080\n"
        "Net investment income | 660 | 21 | 20"
    )
    assert relevance_score(chunk, evidence) >= 0.5


def test_relevant_chunk_ids() -> None:
    chunks = [{"chunk_id": "match", "text": HTML_CHUNK}, {"chunk_id": "other", "text": "x " * 50}]
    assert relevant_chunk_ids(chunks, [PDF_EVIDENCE]) == {"match"}


@pytest.mark.parametrize(
    ("ranked", "expected_hit1", "expected_hit5", "expected_rr"),
    [
        (["x", "rel", "y"], 0.0, 1.0, 0.5),
        (["rel", "x"], 1.0, 1.0, 1.0),
        (["x", "y"], 0.0, 0.0, 0.0),
    ],
)
def test_hit_and_reciprocal_rank(
    ranked: list[str], expected_hit1: float, expected_hit5: float, expected_rr: float
) -> None:
    relevant = {"rel"}
    assert hit_at_k(ranked, relevant, 1) == expected_hit1
    assert hit_at_k(ranked, relevant, 5) == expected_hit5
    assert reciprocal_rank(ranked, relevant, 10) == expected_rr


def test_ndcg_with_two_relevant() -> None:
    ranked = ["a", "x", "b"]
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, {"a", "b"}, 10) == pytest.approx(expected)
    assert ndcg_at_k(["x"], set(), 10) == 0.0
