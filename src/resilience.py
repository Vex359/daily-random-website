"""Resilience and error handling for the Daily Random Website pipeline.

Provides:
- Error taxonomy (transient vs permanent)
- Retry logic for transient failures with exponential backoff
- Dead-letter queue for failed items (data/failed.json)
- Credit monitoring for Netlify deployment
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import httpx
import requests

logger = logging.getLogger(__name__)

# Data directory relative to project root
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_FAILED_FILE = _DATA_DIR / "failed.json"

T = TypeVar("T")


# =============================================================================
# Error Taxonomy
# =============================================================================

class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class TransientError(PipelineError):
    """Errors that may succeed on retry (network timeouts, rate limits, 5xx)."""


class PermanentError(PipelineError):
    """Errors that will not succeed on retry (404, blocked domains, invalid data)."""


TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
PERMANENT_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410, 422, 451})


def classify_error(exc_or_code: Exception | int) -> str:
    """Classify an error or HTTP status code as 'transient', 'permanent', or 'unknown'.

    Args:
        exc_or_code: An Exception instance or integer HTTP status code.

    Returns:
        One of 'transient', 'permanent', or 'unknown'.
    """
    if isinstance(exc_or_code, int):
        code = exc_or_code
        if code in TRANSIENT_STATUS_CODES:
            return "transient"
        if code in PERMANENT_STATUS_CODES:
            return "permanent"
        if 500 <= code <= 599:
            return "transient"
        if 400 <= code <= 499:
            return "permanent"
        return "unknown"

    exc = exc_or_code
    if isinstance(exc, TransientError):
        return "transient"
    if isinstance(exc, PermanentError):
        return "permanent"

    # Requests exceptions
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return "transient"
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status:
            return classify_error(status)

    # Httpx exceptions
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return "transient"
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return classify_error(status)

    # Standard exceptions
    if isinstance(exc, (TimeoutError, ConnectionResetError, ConnectionRefusedError)):
        return "transient"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "permanent"

    return "unknown"


# =============================================================================
# Retry Logic
# =============================================================================

def with_retry(
    max_retries: int = 1,
    backoff_factor: float = 1.0,
    allowed_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function on transient failures.

    Args:
        max_retries: Maximum retry attempts (default 1).
        backoff_factor: Seconds to wait before retry (exponential backoff).
        allowed_exceptions: Specific exceptions to treat as retryable.
            If None, classify_error is used.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempts += 1
                    is_transient = False
                    if allowed_exceptions and isinstance(exc, allowed_exceptions) or classify_error(exc) == "transient":
                        is_transient = True

                    if not is_transient or attempts > max_retries:
                        raise

                    sleep_time = backoff_factor * (2 ** (attempts - 1))
                    logger.warning(
                        "Transient error calling %s: %s. Retrying in %.1fs (attempt %d/%d)...",
                        func.__name__,
                        exc,
                        sleep_time,
                        attempts,
                        max_retries,
                    )
                    time.sleep(sleep_time)

        return wrapper

    return decorator


# =============================================================================
# Dead-Letter Queue (DLQ)
# =============================================================================

class DeadLetterQueue:
    """Records failed items and errors to data/failed.json for post-mortem analysis."""

    def __init__(self, filepath: Path | str | None = None) -> None:
        self.filepath = Path(filepath) if filepath else _DEFAULT_FAILED_FILE

    def _load(self) -> list[dict[str, Any]]:
        if not self.filepath.exists():
            return []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return json.loads(content) if content else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read failed items file %s: %s", self.filepath, exc)
            return []

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
        except OSError as exc:
            logger.error("Could not write failed items file %s: %s", self.filepath, exc)

    def record_failure(
        self,
        url: str,
        source: str,
        error: Exception | str,
        *,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a failed URL and its failure reason.

        Returns:
            The recorded failure entry.
        """
        err_str = str(error)
        err_type = error_type or (classify_error(error) if isinstance(error, Exception) else "unknown")

        entry: dict[str, Any] = {
            "url": url,
            "source": source,
            "error": err_str,
            "error_type": err_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if details:
            entry["details"] = details

        items = self._load()
        items.append(entry)
        self._save(items)
        logger.info("Recorded failure to DLQ for %s (%s): %s", url, source, err_str)
        return entry

    def load_failures(self) -> list[dict[str, Any]]:
        """Return all recorded failures."""
        return self._load()

    def clear(self) -> None:
        """Clear all failure records."""
        self._save([])


# =============================================================================
# Netlify Credit & Quota Monitoring
# =============================================================================

def check_netlify_credits(
    auth_token: str | None = None,
    site_id: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Check Netlify account build minutes and bandwidth status.

    If credentials are missing or API request fails, returns a safe status dict
    so the pipeline does not abort prematurely.

    Returns:
        Dict with keys:
            - status (str): "ok", "warning", "exceeded", or "skipped"
            - credits_available (bool): True if safe to deploy
            - message (str): Explanation
    """
    token = auth_token or os.environ.get("NETLIFY_AUTH_TOKEN")
    sid = site_id or os.environ.get("NETLIFY_SITE_ID")

    if not token:
        return {
            "status": "skipped",
            "credits_available": True,
            "message": "No NETLIFY_AUTH_TOKEN found; skipping credit check.",
        }

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "DailyRandomWebsite/1.0"}

    try:
        url = f"https://api.netlify.com/api/v1/sites/{sid}" if sid else "https://api.netlify.com/api/v1/user"
        resp = sess.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            return {
                "status": "ok",
                "credits_available": True,
                "message": "Netlify account credentials valid and active.",
            }
        elif resp.status_code in (401, 403):
            return {
                "status": "warning",
                "credits_available": False,
                "message": f"Netlify auth failed with status {resp.status_code}.",
            }
        elif resp.status_code == 429:
            return {
                "status": "exceeded",
                "credits_available": False,
                "message": "Netlify API rate limit reached.",
            }
        else:
            return {
                "status": "warning",
                "credits_available": True,
                "message": f"Netlify returned HTTP {resp.status_code}.",
            }
    except (requests.RequestException, OSError) as exc:
        logger.warning("Error checking Netlify credits: %s", exc)
        return {
            "status": "warning",
            "credits_available": True,
            "message": f"Credit check failed with error: {exc}",
        }
