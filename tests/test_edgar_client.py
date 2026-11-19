from typing import Any

import pytest
import requests

from finrag.ingest.edgar_client import EdgarClient


class StubResponse:
    def __init__(self, status_code: int, payload: Any = None, headers: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}
        self.content = b"body"

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class StubSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.headers: dict[str, str] = {}
        self.outcomes = list(outcomes)
        self.requested_urls: list[str] = []

    def get(self, url: str, timeout: float) -> StubResponse:
        self.requested_urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes: list[Any], sleeps: list[float]) -> tuple[EdgarClient, StubSession]:
    session = StubSession(outcomes)
    client = EdgarClient(
        "Test User test@example.com", session=session, max_retries=2, sleep=sleeps.append
    )
    return client, session


def test_rejects_user_agent_without_contact() -> None:
    with pytest.raises(ValueError, match="User-Agent"):
        EdgarClient("", session=StubSession([]))


def test_sets_user_agent_header() -> None:
    client, session = make_client([StubResponse(200, {"ok": True})], [])
    assert client.get_json("https://example.test") == {"ok": True}
    assert session.headers["User-Agent"] == "Test User test@example.com"


def test_retries_rate_limited_response_honouring_retry_after() -> None:
    sleeps: list[float] = []
    client, session = make_client(
        [StubResponse(429, headers={"Retry-After": "3"}), StubResponse(200, {"ok": True})], sleeps
    )
    assert client.get_json("https://example.test") == {"ok": True}
    assert 3.0 in sleeps
    assert len(session.requested_urls) == 2


def test_retries_connection_errors_then_gives_up() -> None:
    client, _ = make_client([requests.ConnectionError("down")] * 3, [])
    with pytest.raises(requests.ConnectionError):
        client.get_bytes("https://example.test")


def test_non_retryable_error_raises_immediately() -> None:
    client, session = make_client([StubResponse(404)], [])
    with pytest.raises(requests.HTTPError):
        client.get_bytes("https://example.test")
    assert len(session.requested_urls) == 1
