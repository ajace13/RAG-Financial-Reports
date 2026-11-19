"""OpenAI generator: strict JSON-schema output with verified quotes (see generate/structured.py)."""

import json
from collections.abc import Sequence
from typing import Any

import openai

from finrag.generate.models import Answer, AnswerSegment, GenerationError, Passage
from finrag.generate.prompts import STRUCTURED_SYSTEM_PROMPT
from finrag.generate.structured import ANSWER_SCHEMA, build_structured_input, structured_answer


def find_refusal(response: Any) -> str | None:
    """Return refusal text if any output message content item is a refusal."""
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                return getattr(content, "refusal", "") or "refused"
    return None


class OpenAIGenerator:
    provider = "openai"

    def __init__(self, api_key: str, model: str, client: Any | None = None) -> None:
        self.client = client if client is not None else openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=STRUCTURED_SYSTEM_PROMPT,
                input=build_structured_input(question, passages),
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
        return structured_answer(
            payload,
            passages,
            provider=self.provider,
            model=getattr(response, "model", self.model),
            stop_reason=getattr(response, "status", "completed") or "completed",
            usage=usage_counts,
        )
