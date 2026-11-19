"""Structured-output citations shared by providers without native citation support.

The model returns JSON where each claim names its passages and quotes them. Quotes are then
checked against the passage text (whitespace- and case-insensitive), and any quote that
isn't actually in the passage is flagged unverified rather than trusted. Used by both the
OpenAI and Ollama generators.
"""

import logging
import re
from collections.abc import Sequence
from typing import Any

from finrag.generate.models import Answer, AnswerSegment, Citation, Passage
from finrag.generate.prompts import (
    render_passages_for_text_prompt,
    render_question,
    short_passage_id,
)

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answerable", "claims"],
    "properties": {
        "answerable": {"type": "boolean"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "citations"],
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["passage_id", "quote"],
                            "properties": {
                                "passage_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}


def normalize_for_match(text: str) -> str:
    """Collapse whitespace and unify the table cell separator so quotes survive reformatting."""
    return WHITESPACE_PATTERN.sub(" ", text.replace("|", " ")).strip().lower()


def quote_in_passage(quote: str, passage: Passage) -> bool:
    normalized_quote = normalize_for_match(quote)
    return bool(normalized_quote) and normalized_quote in normalize_for_match(passage.text)


def build_structured_input(question: str, passages: Sequence[Passage]) -> str:
    """User message: passages labelled P1..Pn, then the question."""
    return (
        f"Passages:\n\n{render_passages_for_text_prompt(passages)}\n\n{render_question(question)}"
    )


def parse_structured_segments(
    payload: dict[str, Any], passages: Sequence[Passage]
) -> list[AnswerSegment]:
    """Turn JSON claims into answer segments, verifying each quote against its passage."""
    passages_by_short_id = {
        short_passage_id(index): passage for index, passage in enumerate(passages)
    }
    segments = []
    claims = payload.get("claims", [])
    for position, claim in enumerate(claims):
        citations = []
        for raw in claim.get("citations", []):
            passage = passages_by_short_id.get(raw.get("passage_id", ""))
            quote = raw.get("quote", "")
            if passage is None:
                logger.warning("Model cited unknown passage id %r", raw.get("passage_id"))
                continue
            citations.append(
                Citation(
                    passage_id=passage.passage_id,
                    cited_text=quote,
                    verified=quote_in_passage(quote, passage),
                )
            )
        separator = " " if position < len(claims) - 1 else ""
        segments.append(
            AnswerSegment(text=claim.get("text", "") + separator, citations=tuple(citations))
        )
    return segments


def structured_answer(
    payload: dict[str, Any],
    passages: Sequence[Passage],
    provider: str,
    model: str,
    stop_reason: str,
    usage: dict[str, int],
) -> Answer:
    """Build an Answer from a parsed JSON payload, with notes for unanswerable/unverified."""
    answer = Answer(
        segments=parse_structured_segments(payload, passages),
        provider=provider,
        model=model,
        stop_reason=stop_reason,
        usage=usage,
    )
    if not payload.get("answerable", True):
        answer.notes.append("The model reported that the passages do not contain the answer.")
    unverified = [citation for citation in answer.citations if not citation.verified]
    if unverified:
        answer.notes.append(
            f"{len(unverified)} quote(s) could not be found verbatim in the cited passage."
        )
    return answer
