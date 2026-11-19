"""OpenAI generator: structured-output citations plus verbatim quote verification.

Without a native citation feature, we ask for JSON where each claim names its passages and
quotes them. Quotes are checked against the passage text after normalizing whitespace, and
a quote that doesn't appear in the passage is flagged unverified rather than trusted.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

import openai

from finrag.generate.models import Answer, AnswerSegment, Citation, GenerationError, Passage
from finrag.generate.prompts import (
    OPENAI_SYSTEM_PROMPT,
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


def build_input(question: str, passages: Sequence[Passage]) -> str:
    return (
        f"Passages:\n\n{render_passages_for_text_prompt(passages)}\n\n{render_question(question)}"
    )


def find_refusal(response: Any) -> str | None:
    """Return refusal text if any output message content item is a refusal."""
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                return getattr(content, "refusal", "") or "refused"
    return None


def parse_structured_answer(
    payload: dict[str, Any], passages: Sequence[Passage]
) -> list[AnswerSegment]:
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


class OpenAIGenerator:
    provider = "openai"

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self.client = client if client is not None else openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=OPENAI_SYSTEM_PROMPT,
                input=build_input(question, passages),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_answer",
                        "schema": ANSWER_SCHEMA,
                        "strict": True,
                    }
                },
            )
        except openai.APIStatusError as error:
            raise GenerationError(
                f"OpenAI API error {error.status_code}: {error.message}"
            ) from error
        except openai.APIConnectionError as error:
            raise GenerationError(f"Could not reach the OpenAI API: {error}") from error

        usage = getattr(response, "usage", None)
        usage_counts = {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        }
        refusal = find_refusal(response)
        if refusal is not None:
            return Answer(
                segments=[AnswerSegment(text=refusal)],
                provider=self.provider,
                model=self.model,
                stop_reason="refusal",
                refused=True,
                usage=usage_counts,
                notes=["The model declined to answer."],
            )
        try:
            payload = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as error:
            raise GenerationError(
                f"OpenAI returned non-JSON output (status {getattr(response, 'status', '?')})"
            ) from error

        answer = Answer(
            segments=parse_structured_answer(payload, passages),
            provider=self.provider,
            model=getattr(response, "model", self.model),
            stop_reason=getattr(response, "status", "completed") or "completed",
            usage=usage_counts,
        )
        if not payload.get("answerable", True):
            answer.notes.append("The model reported that the passages do not contain the answer.")
        unverified = [citation for citation in answer.citations if not citation.verified]
        if unverified:
            answer.notes.append(
                f"{len(unverified)} quote(s) could not be found verbatim in the cited passage."
            )
        return answer
