"""Runtime configuration read from the process environment and an optional .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ES_URL = "http://localhost:9200"
DEFAULT_ES_INDEX = "finrag-chunks"
DEFAULT_DATA_DIR = "data"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# Chosen from the FinanceBench ablation: ms-marco-MiniLM-L-6-v2 lowered Hit@10 (it struggles
# with table passages); bge-reranker-base raised it, at ~600 ms per query.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_EFFORT = "high"
DEFAULT_OPENAI_MODEL = "gpt-6.2-astra"


@dataclass(frozen=True)
class Settings:
    """Typed view of the project's configuration."""

    es_url: str
    es_index: str
    sec_user_agent: str | None
    data_dir: Path
    embedding_model: str
    reranker_model: str
    anthropic_api_key: str | None
    claude_model: str
    claude_effort: str
    openai_api_key: str | None
    openai_model: str


def resolve_data_dir(raw_value: str) -> Path:
    """Resolve a DATA_DIR value. Relative paths are anchored at the repo root, not the CWD."""
    path = Path(raw_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_settings(env_file: Path | None = None) -> Settings:
    """Build Settings from the process environment, falling back to values in `env_file`.

    `env_file` defaults to `<repo root>/.env`. A missing file is fine. Process environment
    variables take precedence over the file, and empty values count as unset.
    """
    file_values = dotenv_values(env_file or PROJECT_ROOT / ".env")

    def lookup(name: str, default: str) -> str:
        return os.environ.get(name) or file_values.get(name) or default

    sec_user_agent = lookup("SEC_USER_AGENT", "").strip() or None
    return Settings(
        es_url=lookup("ES_URL", DEFAULT_ES_URL),
        es_index=lookup("ES_INDEX", DEFAULT_ES_INDEX),
        sec_user_agent=sec_user_agent,
        data_dir=resolve_data_dir(lookup("DATA_DIR", DEFAULT_DATA_DIR)),
        embedding_model=lookup("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        reranker_model=lookup("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        anthropic_api_key=lookup("ANTHROPIC_API_KEY", "").strip() or None,
        claude_model=lookup("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
        claude_effort=lookup("CLAUDE_EFFORT", DEFAULT_CLAUDE_EFFORT),
        openai_api_key=lookup("OPENAI_API_KEY", "").strip() or None,
        openai_model=lookup("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    )
