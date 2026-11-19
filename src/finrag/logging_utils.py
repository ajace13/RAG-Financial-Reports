"""Shared logging setup for command-line scripts."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a compact timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
