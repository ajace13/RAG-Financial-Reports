"""Ollama generator with a fake HTTP session (no Ollama server needed)."""

import json
from typing import Any

import pytest
import requests

from finrag.generate.models import GenerationError, Passage
from finrag.generate.ollama import OllamaGenerator
from finrag.generate.structured import ANSWER_SCHEMA

CAPEX_ROW = "Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)"
PASSAGE = Passage(
    passage_id="acc-00156",
    label="3M CO (MMM) 10-K FY2018 | Item 7. MD&A (in millions)",
    source_url="https://www.sec.gov/x",
    text=f"(Millions) | 2018 | 2017\n{CAPEX_ROW}",
    blocks=("(Millions) | 2018 | 2017", CAPEX_ROW),
)
GOOD_PAYLOAD = {
    "answerable": True,
    "claims": [
        {
            "text": "3M's capital expenditure was $1,577 million in fiscal 2018.",
            "citations": [
                {
                    "passage_id": "P1",
                    "quote": "Purchases of property, plant and equipment (PP&E) | $(1,577)",
                }
            ],
        }
    ],
}


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.body = body or {}
        self.text = text

    def json(self) -> dict:
        return self.body


class FakeSession:
    """Returns (or raises) one outcome per call; the last outcome repeats."""

    def __init__(self, outcome: Any) -> None:
        self.outcomes = list(outcome) if isinstance(outcome, list) else [outcome]
        self.headers: dict[str, str] = {}
        self.requests: list[dict] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.requests.append({"url": url, "json": dict(json)})
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chat_body(payload: dict, prompt_tokens: int = 1200, done_reason: str = "stop") -> dict:
    return {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": json.dumps(payload)},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_tokens,
        "eval_count": 80,
    }


def make_generator(outcome: Any, **kwargs: Any) -> tuple[OllamaGenerator, FakeSession]:
    session = FakeSession(outcome)
    generator = OllamaGenerator(
        "http://localhost:11434/", "qwen3:8b", num_ctx=8192, session=session, **kwargs
    )
    return generator, session


def test_request_sets_schema_context_and_temperature() -> None:
    generator, session = make_generator(FakeResponse(200, chat_body(GOOD_PAYLOAD)))
    answer = generator.generate("What was capex?", [PASSAGE])
    sent = session.requests[0]
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["json"]["format"] == ANSWER_SCHEMA
    assert sent["json"]["options"] == {"num_ctx": 8192, "temperature": 0}
    assert sent["json"]["stream"] is False
    assert "think" not in sent["json"]  # model default unless configured
    assert "[P1] 3M CO (MMM)" in sent["json"]["messages"][1]["content"]
    assert answer.is_grounded
    assert answer.cited_passage_ids == ["acc-00156"]
    assert answer.usage == {"input_tokens": 1200, "output_tokens": 80}
    assert answer.notes == []


def test_think_and_api_key_are_forwarded_when_configured() -> None:
    generator, session = make_generator(
        FakeResponse(200, chat_body(GOOD_PAYLOAD)), think=False, api_key="cloud-key"
    )
    generator.generate("q", [PASSAGE])
    assert session.requests[0]["json"]["think"] is False
    assert session.headers["Authorization"] == "Bearer cloud-key"


def test_near_full_context_and_length_stop_add_notes() -> None:
    body = chat_body(GOOD_PAYLOAD, prompt_tokens=8150, done_reason="length")
    generator, _ = make_generator(FakeResponse(200, body))
    notes = generator.generate("q", [PASSAGE]).notes
    assert any("cut off" in note for note in notes)
    assert any("OLLAMA_NUM_CTX" in note for note in notes)


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (requests.ConnectionError("refused"), "Is it running"),
        (FakeResponse(404, text="model not found"), "ollama pull qwen3:8b"),
        (FakeResponse(500, text="boom"), "Ollama error 500"),
        (
            FakeResponse(200, {"message": {"content": "not json"}, "done_reason": "stop"}),
            "non-JSON",
        ),
    ],
)
def test_failures_become_generation_errors_with_actionable_messages(
    outcome: Any, message: str
) -> None:
    generator, _ = make_generator(outcome)
    with pytest.raises(GenerationError, match=message):
        generator.generate("q", [PASSAGE])


def test_model_without_thinking_mode_is_retried_without_think() -> None:
    rejected = FakeResponse(400, text='{"error":"\\"llama3.1:8b\\" does not support thinking"}')
    generator, session = make_generator(
        [rejected, FakeResponse(200, chat_body(GOOD_PAYLOAD))], think=False
    )
    answer = generator.generate("q", [PASSAGE])
    assert answer.is_grounded
    assert session.requests[0]["json"]["think"] is False
    assert "think" not in session.requests[1]["json"]
    assert generator.think is None  # later requests skip the field too
