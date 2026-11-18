"""Verify that the configured Elasticsearch cluster is reachable and healthy."""

import sys

from finrag.config import load_settings
from finrag.index.client import make_client, wait_for_cluster


def main() -> int:
    """Wait for the cluster, print its version and health, and return a process exit code."""
    settings = load_settings()
    client = make_client(settings)
    print(f"Connecting to {settings.es_url} ...")
    try:
        health = wait_for_cluster(client, timeout_seconds=90)
    except TimeoutError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    version = client.info()["version"]["number"]
    cluster_name = health["cluster_name"]
    print(f"OK: Elasticsearch {version}, cluster '{cluster_name}', status={health['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
