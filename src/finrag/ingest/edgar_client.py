"""Minimal HTTP client for SEC EDGAR that follows the SEC fair-access policy.

https://www.sec.gov/os/accessing-edgar-data: identify yourself with a User-Agent that
includes contact info and stay under 10 requests per second.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_TIMEOUT_SECONDS = 30


class EdgarClient:
    """Rate-limited, retrying GET client for sec.gov and data.sec.gov."""

    def __init__(
        self,
        user_agent: str,
        session: requests.Session | None = None,
        max_requests_per_second: float = 8.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent with contact info, e.g. 'Jane Doe jane@example.com'. "
                "Set SEC_USER_AGENT in .env."
            )
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval_seconds = 1.0 / max_requests_per_second
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self._last_request_at: float | None = None

    def get_json(self, url: str) -> Any:
        """GET a URL and decode the JSON body."""
        return self._get(url).json()

    def get_bytes(self, url: str) -> bytes:
        """GET a URL and return the raw body."""
        return self._get(url).content

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval_seconds:
                self.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def _retry_delay(self, attempt: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                return float(retry_after)
        return self.backoff_seconds * 2**attempt

    def _get(self, url: str) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            except (requests.ConnectionError, requests.Timeout) as error:
                if attempt == self.max_retries:
                    raise
                delay = self._retry_delay(attempt, None)
                logger.warning("GET %s failed (%s); retrying in %.1fs", url, error, delay)
                self.sleep(delay)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                delay = self._retry_delay(attempt, response)
                logger.warning(
                    "GET %s returned %s; retrying in %.1fs", url, response.status_code, delay
                )
                self.sleep(delay)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError("unreachable: retry loop exited without returning")
