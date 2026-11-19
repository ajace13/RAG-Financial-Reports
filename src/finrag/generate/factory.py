"""Choose a generator from configuration."""

from finrag.config import Settings
from finrag.generate.claude import ClaudeGenerator
from finrag.generate.fallback import FallbackGenerator
from finrag.generate.models import Generator
from finrag.generate.ollama import OllamaGenerator
from finrag.generate.openai_structured import OpenAIGenerator

PROVIDERS = ("auto", "claude", "openai", "ollama", "none")
REQUIREMENTS = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_MODEL",
}


def configured_generators(settings: Settings) -> dict[str, Generator]:
    """Every provider whose configuration is present, keyed by provider name."""
    generators: dict[str, Generator] = {}
    if settings.anthropic_api_key:
        generators["claude"] = ClaudeGenerator(
            settings.anthropic_api_key, settings.claude_model, settings.claude_effort
        )
    if settings.openai_api_key:
        generators["openai"] = OpenAIGenerator(settings.openai_api_key, settings.openai_model)
    if settings.ollama_model:
        generators["ollama"] = OllamaGenerator(
            settings.ollama_url,
            settings.ollama_model,
            settings.ollama_num_ctx,
            api_key=settings.ollama_api_key,
            think=settings.ollama_think,
        )
    return generators


def build_generator(settings: Settings, provider: str = "auto") -> Generator | None:
    """Return a generator, or None for retrieval-only mode.

    'auto' chains every configured provider in the order Claude -> OpenAI -> Ollama, so a
    failing provider (bad key, no credit, server down) falls through to the next one.
    Naming a specific provider that isn't configured is an error.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if provider == "none":
        return None
    available = configured_generators(settings)
    if provider != "auto":
        if provider not in available:
            raise ValueError(f"provider '{provider}' needs {REQUIREMENTS[provider]} to be set")
        return available[provider]
    chain = [available[name] for name in ("claude", "openai", "ollama") if name in available]
    if not chain:
        return None
    return chain[0] if len(chain) == 1 else FallbackGenerator(chain)
