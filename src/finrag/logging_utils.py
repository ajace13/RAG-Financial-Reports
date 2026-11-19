"""Shared logging setup for command-line scripts."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a compact timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # The ES transport logs every HTTP request at INFO; keep warnings and errors only.
    for noisy_logger in (
        "elastic_transport",
        "httpx",
        "httpx2",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
