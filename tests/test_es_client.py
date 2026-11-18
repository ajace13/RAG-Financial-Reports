from collections.abc import Iterable, Iterator
from itertools import repeat
from typing import Any

import pytest
from elasticsearch import ConnectionError as ESConnectionError

from finrag.config import load_settings
from finrag.index.client import make_client, wait_for_cluster


class StubCluster:
    """Returns (or raises) queued health responses in order."""

    def __init__(self, responses: Iterable[Any]) -> None:
        self.responses: Iterator[Any] = iter(responses)

    def health(self) -> dict[str, str]:
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class StubClient:
    def __init__(self, responses: Iterable[Any]) -> None:
        self.cluster = StubCluster(responses)


def test_returns_once_cluster_turns_yellow() -> None:
    client = StubClient([ESConnectionError("refused"), {"status": "red"}, {"status": "yellow"}])
    health = wait_for_cluster(client, timeout_seconds=5, poll_seconds=0)
    assert health["status"] == "yellow"


def test_times_out_when_cluster_stays_red() -> None:
    client = StubClient(repeat({"status": "red"}))
    with pytest.raises(TimeoutError, match="last status: red"):
        wait_for_cluster(client, timeout_seconds=0.05, poll_seconds=0.01)


def test_timeout_chains_last_connection_error() -> None:
    client = StubClient(repeat(ESConnectionError("refused")))
    with pytest.raises(TimeoutError) as excinfo:
        wait_for_cluster(client, timeout_seconds=0.05, poll_seconds=0.01)
    assert isinstance(excinfo.value.__cause__, ESConnectionError)


@pytest.mark.integration
def test_live_cluster_is_healthy() -> None:
    client = make_client(load_settings())
    health = wait_for_cluster(client, timeout_seconds=10)
    assert health["status"] in ("yellow", "green")
