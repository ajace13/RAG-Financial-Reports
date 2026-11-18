"""Elasticsearch client construction and readiness checks."""

import time
from collections.abc import Mapping
from typing import Any

from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import Elasticsearch

from finrag.config import Settings

HEALTHY_STATUSES = ("yellow", "green")


def make_client(settings: Settings) -> Elasticsearch:
    """Create an Elasticsearch client for the configured cluster."""
    return Elasticsearch(settings.es_url, request_timeout=30)


def wait_for_cluster(
    client: Elasticsearch, timeout_seconds: float = 60.0, poll_seconds: float = 2.0
) -> Mapping[str, Any]:
    """Poll cluster health until it is yellow or green, then return the health response.

    Connection errors are retried, because the container accepts connections a few seconds
    after it starts. Raises TimeoutError if the cluster is not healthy within `timeout_seconds`.
    """
    deadline = time.monotonic() + timeout_seconds
    last_status = "unreachable"
    last_error: Exception | None = None
    while True:
        try:
            health = client.cluster.health()
            last_status = health["status"]
            if last_status in HEALTHY_STATUSES:
                return health
        except ESConnectionError as error:
            last_error = error
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Elasticsearch not healthy after {timeout_seconds:.0f}s (last status: {last_status})"
    ) from last_error
