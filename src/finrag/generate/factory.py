"""Choose a generator from configured credentials."""

from finrag.config import Settings
from finrag.generate.claude import ClaudeGenerator
from finrag.generate.fallback import FallbackGenerator
from finrag.generate.models import Generator
from finrag.generate.openai_structured import OpenAIGenerator

PROVIDERS = ("auto", "claude", "openai", "none")


def build_generator(settings: Settings, provider: str = "auto") -> Generator | None:
    """Return a generator, or None for retrieval-only mode.

    'auto' uses Claude when ANTHROPIC_API_KEY is set, OpenAI when OPENAI_API_KEY is set, and
    Claude-then-OpenAI when both are. Naming a provider without its key is an error.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    if provider == "none":
        return None

    claude = (
        ClaudeGenerator(settings.anthropic_api_key, settings.claude_model, settings.claude_effort)
        if settings.anthropic_api_key and provider in ("auto", "claude")
        else None
    )
    openai_generator = (
        OpenAIGenerator(settings.openai_api_key, settings.openai_model)
        if settings.openai_api_key and provider in ("auto", "openai")
        else None
    )
    if provider == "claude" and claude is None:
        raise ValueError("provider 'claude' needs ANTHROPIC_API_KEY")
    if provider == "openai" and openai_generator is None:
        raise ValueError("provider 'openai' needs OPENAI_API_KEY")

    available = [generator for generator in (claude, openai_generator) if generator is not None]
    if not available:
        return None
    return available[0] if len(available) == 1 else FallbackGenerator(available)
