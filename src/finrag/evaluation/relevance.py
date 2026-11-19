"""Decide which chunks are relevant to a FinanceBench question by matching its evidence text.

FinanceBench gives evidence as text extracted from PDF pages, while our chunks come from
EDGAR HTML. Formatting differs (line breaks, "$" placement, and extraction artifacts such as
"Cash Flow s" or words run together), so we compare normalized n-grams, not strings.

Evidence ranges from one sentence to a whole page, so containment is checked both ways:
- chunk inside evidence: the chunk was drawn from the evidence page (long evidence);
- evidence inside chunk: the evidence sentence appears in the chunk (short evidence).
Each check runs on word+number tokens and also on numbers alone. Numbers survive PDF
extraction even when spacing between words is lost, which matters most for tables.
"""

import re
from collections.abc import Iterable

TOKEN_PATTERN = re.compile(r"[a-z]+|\d[\d,]*(?:\.\d+)?")
DEFAULT_NGRAM = 3
DEFAULT_RELEVANCE_THRESHOLD = 0.5
# The contained side needs at least this many n-grams; very short texts match by accident.
MIN_INNER_NGRAMS = 8


def normalize_tokens(text: str) -> list[str]:
    """Lowercased words and numbers; thousands separators removed so '1,577' == '1577'."""
    return [token.replace(",", "") for token in TOKEN_PATTERN.findall(text.lower())]


def number_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token[0].isdigit()]


def ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def ngram_containment(inner: list[str], outer: list[str], size: int = DEFAULT_NGRAM) -> float:
    """Fraction of `inner`'s n-grams that also occur in `outer` (0 if `inner` is too short)."""
    inner_ngrams = ngrams(inner, size)
    if len(inner_ngrams) < MIN_INNER_NGRAMS:
        return 0.0
    return len(inner_ngrams & ngrams(outer, size)) / len(inner_ngrams)


def relevance_score(chunk_text: str, evidence_text: str) -> float:
    """Best containment across both directions, on all tokens and on numbers only."""
    chunk_tokens = normalize_tokens(chunk_text)
    evidence_tokens = normalize_tokens(evidence_text)
    chunk_numbers = number_tokens(chunk_tokens)
    evidence_numbers = number_tokens(evidence_tokens)
    return max(
        ngram_containment(chunk_tokens, evidence_tokens),
        ngram_containment(evidence_tokens, chunk_tokens),
        ngram_containment(chunk_numbers, evidence_numbers),
        ngram_containment(evidence_numbers, chunk_numbers),
    )


def relevant_chunk_ids(
    chunks: Iterable[dict],
    evidence_texts: Iterable[str],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> set[str]:
    """IDs of chunks that match any of the evidence passages."""
    evidence_list = list(evidence_texts)
    return {
        chunk["chunk_id"]
        for chunk in chunks
        if any(relevance_score(chunk["text"], evidence) >= threshold for evidence in evidence_list)
    }
