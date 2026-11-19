from pathlib import Path

import pytest

from finrag.config import (
    DEFAULT_ES_INDEX,
    DEFAULT_ES_URL,
    PROJECT_ROOT,
    load_settings,
)

CONFIG_VARS = (
    "ES_URL",
    "ES_INDEX",
    "SEC_USER_AGENT",
    "DATA_DIR",
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)


def write_env_file(directory: Path, contents: str) -> Path:
    env_file = directory / "test.env"
    env_file.write_text(contents, encoding="utf-8")
    return env_file


def test_defaults_when_nothing_is_set(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.env")
    assert settings.es_url == DEFAULT_ES_URL
    assert settings.es_index == DEFAULT_ES_INDEX
    assert settings.sec_user_agent is None
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.reranker_model == "BAAI/bge-reranker-base"


def test_values_read_from_env_file(tmp_path: Path) -> None:
    env_file = write_env_file(
        tmp_path, "ES_URL=http://es:9200\nSEC_USER_AGENT=Test test@example.com\n"
    )
    settings = load_settings(env_file)
    assert settings.es_url == "http://es:9200"
    assert settings.sec_user_agent == "Test test@example.com"


def test_process_environment_overrides_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env_file(tmp_path, "ES_INDEX=from-file\n")
    monkeypatch.setenv("ES_INDEX", "from-env")
    assert load_settings(env_file).es_index == "from-env"


def test_blank_user_agent_is_treated_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    assert load_settings(tmp_path / "missing.env").sec_user_agent is None


def test_relative_data_dir_is_anchored_at_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", "custom/data")
    assert load_settings(tmp_path / "missing.env").data_dir == PROJECT_ROOT / "custom" / "data"


def test_absolute_data_dir_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert load_settings(tmp_path / "missing.env").data_dir == tmp_path
