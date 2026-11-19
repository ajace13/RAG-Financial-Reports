"""Check which LLM providers are configured and whether they actually work.

For Ollama this checks, in order: server reachable -> model pulled -> one tiny grounded
answer in JSON mode (timed; the first call also loads the model into memory). API keys for
Claude/OpenAI are reported as present or absent, never printed.
"""

import time

import requests

from finrag.config import load_settings
from finrag.generate.factory import configured_generators
from finrag.generate.models import GenerationError, Passage
from finrag.generate.ollama import list_local_models

PROBE_PASSAGE = Passage(
    passage_id="probe-1",
    label="EXAMPLE CORP (EXM) 10-K FY2024 | Item 8. Financial Statements (in millions)",
    source_url="https://example.com",
    text=(
        "Consolidated Statements of Income\n(In millions) | 2024 | 2023\n"
        "Total revenue | $1,250 | $1,100"
    ),
    blocks=(
        "Consolidated Statements of Income",
        "(In millions) | 2024 | 2023",
        "Total revenue | $1,250 | $1,100",
    ),
)
PROBE_QUESTION = "What was Example Corp's total revenue in fiscal 2024?"


def check_ollama_server(url: str, model: str) -> bool:
    """Report server reachability and whether `model` is pulled; True if both are fine."""
    try:
        models = list_local_models(url)
    except requests.RequestException as error:
        print(f"  [FAIL] Ollama server not reachable at {url} ({type(error).__name__}).")
        print("         Start the Ollama app, or run `ollama serve`.")
        return False
    print(f"  [ok]   Ollama server reachable at {url}; {len(models)} model(s) pulled")
    base_names = {name.removesuffix(":latest") for name in models}
    if model not in models and model.removesuffix(":latest") not in base_names:
        print(f"  [FAIL] Model '{model}' not pulled. Run `ollama pull {model}`.")
        print(f"         Pulled models: {', '.join(models) or 'none'}")
        return False
    print(f"  [ok]   Model '{model}' is pulled")
    return True


def main() -> int:
    """Print a status line per provider; exit 1 if nothing usable is configured."""
    settings = load_settings()
    print("Configuration:")
    print(f"  ANTHROPIC_API_KEY: {'set' if settings.anthropic_api_key else 'not set'}")
    print(f"  OPENAI_API_KEY:    {'set' if settings.openai_api_key else 'not set'}")
    print(f"  OLLAMA_MODEL:      {settings.ollama_model or 'not set'}")

    generators = configured_generators(settings)
    if not generators:
        print("\nNo LLM configured: scripts/ask.py will run in retrieval-only mode.")
        print("For free local generation: install Ollama, `ollama pull qwen3:8b`, and set")
        print("OLLAMA_MODEL=qwen3:8b in .env.")
        return 1

    working = 0
    for name, generator in generators.items():
        print(f"\n{name}:")
        if name == "ollama" and not check_ollama_server(settings.ollama_url, settings.ollama_model):
            continue
        started = time.perf_counter()
        try:
            answer = generator.generate(PROBE_QUESTION, [PROBE_PASSAGE])
        except GenerationError as error:
            print(f"  [FAIL] {error}")
            continue
        elapsed = time.perf_counter() - started
        verified = sum(citation.verified for citation in answer.citations)
        print(f"  [ok]   Answered in {elapsed:.1f}s: {answer.text.strip()[:160]}")
        citation_count = len(answer.citations)
        print(f"         citations: {citation_count} ({verified} verified)")
        print(f"         tokens: {answer.usage}")
        for note in answer.notes:
            print(f"         note: {note}")
        working += 1
    print(f"\n{working} of {len(generators)} configured provider(s) working.")
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
