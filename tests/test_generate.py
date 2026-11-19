"""Generation: passages, Claude citation parsing, OpenAI quote verification, fallback, factory.

SDK clients are replaced by fakes, so no API key or network is needed.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finrag.config import load_settings
from finrag.generate.claude import ClaudeGenerator, build_user_content
from finrag.generate.factory import build_generator
from finrag.generate.fallback import FallbackGenerator
from finrag.generate.models import Answer, AnswerSegment, GenerationError, Passage
from finrag.generate.openai_structured import OpenAIGenerator
from finrag.generate.passages import passage_from_hit, split_into_blocks
from finrag.generate.structured import quote_in_passage
from finrag.retrieve.models import SearchHit

TABLE_TEXT = (
    "Cash Flows from Investing Activities:\n(Millions) | 2018 | 2017\n"
    "Purchases of property, plant and equipment (PP&E) | $(1,577) | $(1,373)"
)
DOCUMENT = {
    "company_name": "3M CO",
    "ticker": "MMM",
    "form": "10-K",
    "fiscal_year": 2018,
    "fiscal_period": "FY",
    "item": "7",
    "section_title": "Management's Discussion and Analysis",
    "units": "in millions",
    "chunk_type": "table",
    "text": TABLE_TEXT,
    "source_url": "https://www.sec.gov/Archives/edgar/data/66740/x/mmm-20181231x10k.htm",
}
PASSAGE = passage_from_hit(SearchHit(chunk_id="acc-00156", score=1.0, document=DOCUMENT))
OTHER_PASSAGE = replace(
    PASSAGE, passage_id="acc-00001", text="Unrelated text.", blocks=("Unrelated text.",)
)


def test_passage_label_and_table_blocks() -> None:
    assert PASSAGE.label == (
        "3M CO (MMM) 10-K FY2018 | Item 7. Management's Discussion and Analysis (in millions)"
    )
    assert PASSAGE.blocks[-1].startswith("Purchases of property")
    assert len(PASSAGE.blocks) == 3


def test_prose_splits_into_sentences() -> None:
    assert split_into_blocks("Revenue grew. Margins fell.\nCash rose.", "prose") == (
        "Revenue grew.",
        "Margins fell.",
        "Cash rose.",
    )


def test_claude_request_uses_search_results_with_citations() -> None:
    content = build_user_content("What was capex?", [PASSAGE])
    search_result = content[0]
    assert search_result["type"] == "search_result"
    assert search_result["source"] == "acc-00156"
    assert search_result["citations"] == {"enabled": True}
    assert [block["text"] for block in search_result["content"]] == list(PASSAGE.blocks)
    assert content[-1] == {"type": "text", "text": "Question: What was capex?"}


class FakeClaudeClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def claude_response(stop_reason: str = "end_turn", source: str = "acc-00156") -> Any:
    citation = SimpleNamespace(
        type="search_result_location",
        source=source,
        title=PASSAGE.label,
        cited_text=PASSAGE.blocks[-1],
        search_result_index=0,
        start_block_index=2,
        end_block_index=3,
    )
    return SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=900, output_tokens=120),
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(
                type="text",
                text="3M spent $1,577 million on PP&E in fiscal 2018.",
                citations=[citation],
            ),
            SimpleNamespace(type="text", text=" That is its capital expenditure.", citations=None),
        ],
    )


def test_claude_generator_parses_citations_and_sends_expected_request() -> None:
    client = FakeClaudeClient(claude_response())
    answer = ClaudeGenerator("key", "claude-opus-5-5", "high", client=client).generate(
        "capex?", [PASSAGE]
    )
    assert (
        answer.text
        == "3M spent $1,577 million on PP&E in fiscal 2018. That is its capital expenditure."
    )
    assert answer.cited_passage_ids == ["acc-00156"]
    assert answer.is_grounded
    request = client.calls[0]
    assert request["fallbacks"] == "default"
    assert request["output_config"] == {"effort": "high"}
    assert "thinking" not in request  # adaptive thinking is the model default


def test_claude_mismatched_citation_source_is_unverified() -> None:
    client = FakeClaudeClient(claude_response(source="someone-else"))
    answer = ClaudeGenerator("key", "m", "high", client=client).generate("capex?", [PASSAGE])
    assert not answer.is_grounded


def test_claude_refusal_is_flagged() -> None:
    client = FakeClaudeClient(claude_response(stop_reason="refusal"))
    answer = ClaudeGenerator("key", "m", "high", client=client).generate("q", [PASSAGE])
    assert answer.refused and not answer.is_grounded


class FakeOpenAIClient:
    def __init__(self, payload: dict | None = None, refusal: str | None = None) -> None:
        content = (
            [SimpleNamespace(type="refusal", refusal=refusal)]
            if refusal
            else [SimpleNamespace(type="output_text")]
        )
        self.response = SimpleNamespace(
            model="gpt-6.2-astra",
            status="completed",
            output=[SimpleNamespace(type="message", content=content)],
            output_text=json.dumps(payload) if payload is not None else "",
            usage=SimpleNamespace(input_tokens=800, output_tokens=90),
        )
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def test_openai_quotes_are_verified_against_passage_text() -> None:
    payload = {
        "answerable": True,
        "claims": [
            {
                "text": "3M spent $1,577 million on PP&E in fiscal 2018.",
                "citations": [
                    {
                        "passage_id": "P1",
                        "quote": "Purchases of property, plant and equipment (PP&E)  $(1,577)",
                    },
                ],
            },
            {
                "text": "That was a record.",
                "citations": [{"passage_id": "P2", "quote": "a record year for capex"}],
            },
        ],
    }
    client = FakeOpenAIClient(payload)
    answer = OpenAIGenerator("key", "gpt-6.2-astra", client=client).generate(
        "capex?", [PASSAGE, OTHER_PASSAGE]
    )
    first, second = answer.citations
    assert first.passage_id == "acc-00156" and first.verified
    assert second.passage_id == "acc-00001" and not second.verified
    assert any("could not be found verbatim" in note for note in answer.notes)
    assert client.calls[0]["text"]["format"]["strict"] is True


def test_openai_refusal() -> None:
    client = FakeOpenAIClient(refusal="I can't help with that.")
    answer = OpenAIGenerator("key", "m", client=client).generate("q", [PASSAGE])
    assert answer.refused and answer.text == "I can't help with that."


def test_quote_matching_ignores_pipes_and_whitespace() -> None:
    passage = Passage("id", "label", "url", "Revenue | $245,122 | $211,915", ("x",))
    assert quote_in_passage("Revenue  $245,122", passage)
    assert not quote_in_passage("Revenue $999", passage)
    assert not quote_in_passage("   ", passage)


class StubGenerator:
    def __init__(self, provider: str, fail: bool) -> None:
        self.provider = provider
        self.fail = fail

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        if self.fail:
            raise GenerationError("credit balance too low")
        return Answer(
            segments=[AnswerSegment("ok")],
            provider=self.provider,
            model="m",
            stop_reason="end_turn",
        )


def test_fallback_uses_next_provider_and_notes_why() -> None:
    generator = FallbackGenerator(
        [StubGenerator("claude", fail=True), StubGenerator("openai", fail=False)]
    )
    answer = generator.generate("q", [PASSAGE])
    assert answer.provider == "openai"
    assert "claude: credit balance too low" in answer.notes[0]


def test_fallback_raises_when_all_fail() -> None:
    generator = FallbackGenerator([StubGenerator("claude", fail=True)])
    with pytest.raises(GenerationError, match="All providers failed"):
        generator.generate("q", [PASSAGE])


@pytest.fixture
def no_llm_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OLLAMA_MODEL", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "missing.env"


def test_factory_retrieval_only_without_keys(no_llm_keys: Path) -> None:
    settings = load_settings(no_llm_keys)
    assert build_generator(settings, "auto") is None
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_generator(settings, "claude")


def test_factory_chains_claude_then_openai(
    no_llm_keys: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    generator = build_generator(load_settings(no_llm_keys), "auto")
    assert isinstance(generator, FallbackGenerator)
    assert generator.provider == "claude+openai"
    assert build_generator(load_settings(no_llm_keys), "openai").provider == "openai"


def test_factory_ollama_alone_and_last_in_chain(
    no_llm_keys: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    assert build_generator(load_settings(no_llm_keys), "auto").provider == "ollama"
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    assert build_generator(load_settings(no_llm_keys), "auto").provider == "openai+ollama"
    monkeypatch.delenv("OLLAMA_MODEL")
    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        build_generator(load_settings(no_llm_keys), "ollama")
