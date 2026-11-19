"""Claude generator using search-result content blocks with native citations.

Each passage goes in as a `search_result` block whose `source` is its chunk id. Claude's
citations then point at (passage, block range) and carry the exact cited text, so citations
are verbatim by construction: the API returns spans of our own blocks, never paraphrases.
"""

import logging
from collections.abc import Sequence
from typing import Any

import anthropic

from finrag.generate.models import Answer, AnswerSegment, Citation, GenerationError, Passage
from finrag.generate.prompts import CLAUDE_SYSTEM_PROMPT, render_question

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 16000
# Server-side refusal fallback: if the model declines, the API re-runs the request on a
# fallback model chosen by refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def search_result_block(passage: Passage) -> dict[str, Any]:
    return {
        "type": "search_result",
        "source": passage.passage_id,
        "title": passage.label,
        "content": [{"type": "text", "text": block} for block in passage.blocks],
        "citations": {"enabled": True},
    }


def build_user_content(question: str, passages: Sequence[Passage]) -> list[dict[str, Any]]:
    """Search results first, question last (long context before the instruction)."""
    return [
        *(search_result_block(passage) for passage in passages),
        {"type": "text", "text": render_question(question)},
    ]


def parse_citation(raw: Any, passages: Sequence[Passage]) -> Citation | None:
    """Map a search_result_location citation back to our passage; ignore other citation types."""
    if getattr(raw, "type", None) != "search_result_location":
        return None
    index = raw.search_result_index
    passage = passages[index] if 0 <= index < len(passages) else None
    if passage is None or passage.passage_id != raw.source:
        logger.warning("Citation points at unknown search result %s (%s)", index, raw.source)
        return Citation(passage_id=raw.source, cited_text=raw.cited_text, verified=False)
    return Citation(passage_id=passage.passage_id, cited_text=raw.cited_text, verified=True)


def parse_response(response: Any, passages: Sequence[Passage], provider: str) -> Answer:
    segments = []
    for block in response.content:
        if block.type != "text":
            continue  # thinking and fallback blocks carry no answer text
        citations = [
            parse_citation(raw, passages) for raw in (getattr(block, "citations", None) or [])
        ]
        segments.append(
            AnswerSegment(text=block.text, citations=tuple(c for c in citations if c is not None))
        )
    answer = Answer(
        segments=segments,
        provider=provider,
        model=response.model,
        stop_reason=response.stop_reason or "unknown",
        refused=response.stop_reason == "refusal",
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )
    if response.stop_reason == "max_tokens":
        answer.notes.append("Answer was cut off at the output token limit.")
    if answer.refused:
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        answer.notes.append(f"The model declined to answer (category: {category}).")
    return answer


class ClaudeGenerator:
    provider = "claude"

    def __init__(self, api_key: str, model: str, effort: str, client: Any | None = None) -> None:
        self.client = client if client is not None else anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=CLAUDE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_content(question, passages)}],
            )
        except anthropic.APIStatusError as error:
            raise GenerationError(
                f"Claude API error {error.status_code}: {error.message}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise GenerationError(f"Could not reach the Claude API: {error}") from error
        return parse_response(response, passages, self.provider)
