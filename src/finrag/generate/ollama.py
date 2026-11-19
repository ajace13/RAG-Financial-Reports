"""Ollama generator (local models, free): native /api/chat with a JSON-schema `format`.

Why the native API instead of Ollama's OpenAI-compatible endpoint: the native API takes
`options.num_ctx` per request. Ollama's default context is only 4k tokens on GPUs under
24 GB, and a RAG prompt with 8 passages is about that size. Past the limit the prompt gets
silently truncated, so we set the context window explicitly and warn when a prompt nears it.

Citations use the shared structured-output scheme (generate/structured.py): JSON claims with
quotes, each quote checked verbatim against its passage.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

import requests

from finrag.generate.models import Answer, GenerationError, Passage
from finrag.generate.prompts import STRUCTURED_SYSTEM_PROMPT
from finrag.generate.structured import ANSWER_SCHEMA, build_structured_input, structured_answer

logger = logging.getLogger(__name__)

# Local generation on a laptop GPU can take a while, especially the first call that loads
# the model into memory.
REQUEST_TIMEOUT_SECONDS = 300
# Warn when prompt + output uses this share of the context window.
CONTEXT_WARNING_RATIO = 0.95


class ThinkingUnsupportedError(GenerationError):
    """The model rejected the `think` request field (it has no thinking mode)."""


class OllamaGenerator:
    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        api_key: str | None = None,
        think: bool | None = None,
        session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.session = session if session is not None else requests.Session()
        if api_key:
            # Only needed for Ollama's hosted cloud models; local Ollama has no auth.
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def build_request(self, question: str, passages: Sequence[Passage]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                {"role": "user", "content": build_structured_input(question, passages)},
            ],
            "format": ANSWER_SCHEMA,
            "stream": False,
            # temperature 0: same question, same passages -> same answer (useful for eval).
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        if self.think is not None:  # omit to use the model's default
            request["think"] = self.think
        return request

    def post_chat(self, request: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        try:
            response = self.session.post(url, json=request, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.ConnectionError as error:
            raise GenerationError(
                f"Could not reach Ollama at {self.base_url}. Is it running? "
                "Start the Ollama app (or `ollama serve`)."
            ) from error
        except requests.Timeout as error:
            raise GenerationError(
                f"Ollama took longer than {REQUEST_TIMEOUT_SECONDS}s to answer"
            ) from error
        if response.status_code == 404:
            raise GenerationError(
                f"Ollama model '{self.model}' is not available. Run `ollama pull {self.model}`."
            )
        if response.status_code == 400 and "think" in request and "think" in response.text.lower():
            raise ThinkingUnsupportedError(response.text[:300])
        if response.status_code >= 400:
            raise GenerationError(f"Ollama error {response.status_code}: {response.text[:300]}")
        return response.json()

    def generate(self, question: str, passages: Sequence[Passage]) -> Answer:
        try:
            data = self.post_chat(self.build_request(question, passages))
        except ThinkingUnsupportedError as error:
            # Models without a thinking mode may reject the `think` field; drop it for good.
            logger.warning(
                "%s rejected the think setting (%s); retrying without it", self.model, error
            )
            self.think = None
            data = self.post_chat(self.build_request(question, passages))
        content = (data.get("message") or {}).get("content", "")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise GenerationError(
                f"Ollama returned non-JSON output (done_reason={data.get('done_reason')})"
            ) from error
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0
        answer = structured_answer(
            payload,
            passages,
            provider=self.provider,
            model=data.get("model", self.model),
            stop_reason=data.get("done_reason") or "unknown",
            usage={"input_tokens": prompt_tokens, "output_tokens": output_tokens},
        )
        if data.get("done_reason") == "length":
            answer.notes.append("Answer was cut off at the output length limit.")
        if prompt_tokens + output_tokens >= CONTEXT_WARNING_RATIO * self.num_ctx:
            answer.notes.append(
                f"Prompt and answer used {prompt_tokens + output_tokens} of {self.num_ctx} "
                "context tokens; passages may have been truncated. Raise OLLAMA_NUM_CTX or "
                "pass fewer passages."
            )
        return answer


def list_local_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Names of models pulled into the local Ollama server (GET /api/tags)."""
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]
