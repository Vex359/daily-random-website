"""Unit tests for resilience and error handling module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

from src.resilience import (
    DeadLetterQueue,
    PermanentError,
    TransientError,
    check_netlify_credits,
    classify_error,
    with_retry,
)

# =============================================================================
# Error Taxonomy Tests
# =============================================================================

class TestErrorClassification:
    """Test error taxonomy classification into transient vs permanent."""

    def test_transient_status_codes(self) -> None:
        assert classify_error(408) == "transient"
        assert classify_error(429) == "transient"
        assert classify_error(500) == "transient"
        assert classify_error(502) == "transient"
        assert classify_error(503) == "transient"
        assert classify_error(504) == "transient"

    def test_permanent_status_codes(self) -> None:
        assert classify_error(400) == "permanent"
        assert classify_error(401) == "permanent"
        assert classify_error(403) == "permanent"
        assert classify_error(404) == "permanent"
        assert classify_error(410) == "permanent"
        assert classify_error(422) == "permanent"

    def test_custom_exception_classes(self) -> None:
        assert classify_error(TransientError("test")) == "transient"
        assert classify_error(PermanentError("test")) == "permanent"

    def test_requests_exceptions(self) -> None:
        assert classify_error(requests.Timeout("Timeout")) == "transient"
        assert classify_error(requests.ConnectionError("Failed")) == "transient"

        # HTTPError with status code
        resp = MagicMock()
        resp.status_code = 404
        assert classify_error(requests.HTTPError(response=resp)) == "permanent"

        resp.status_code = 503
        assert classify_error(requests.HTTPError(response=resp)) == "transient"

    def test_httpx_exceptions(self) -> None:
        assert classify_error(httpx.ReadTimeout("Timeout")) == "transient"
        assert classify_error(httpx.ConnectError("Connection refused")) == "transient"

    def test_standard_exceptions(self) -> None:
        assert classify_error(TimeoutError("timed out")) == "transient"
        assert classify_error(ConnectionResetError("reset")) == "transient"
        assert classify_error(ValueError("invalid format")) == "permanent"
        assert classify_error(KeyError("missing key")) == "permanent"


# =============================================================================
# Retry Logic Tests
# =============================================================================

class TestRetryLogic:
    """Test retry decorator with exponential backoff."""

    def test_successful_call_no_retry(self) -> None:
        call_count = 0

        @with_retry(max_retries=2, backoff_factor=0.01)
        def succeeds() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        res = succeeds()
        assert res == "ok"
        assert call_count == 1

    def test_retry_on_transient_error(self) -> None:
        call_count = 0

        @with_retry(max_retries=2, backoff_factor=0.01)
        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.Timeout("Transient network failure")
            return "recovered"

        res = fail_then_succeed()
        assert res == "recovered"
        assert call_count == 2

    def test_no_retry_on_permanent_error(self) -> None:
        call_count = 0

        @with_retry(max_retries=2, backoff_factor=0.01)
        def fail_permanent() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid format")

        with pytest.raises(ValueError):
            fail_permanent()

        assert call_count == 1

    def test_exceed_max_retries(self) -> None:
        call_count = 0

        @with_retry(max_retries=2, backoff_factor=0.01)
        def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise requests.Timeout("Always times out")

        with pytest.raises(requests.Timeout):
            always_fail()

        assert call_count == 3  # initial attempt + 2 retries


# =============================================================================
# Dead-Letter Queue Tests
# =============================================================================

class TestDeadLetterQueue:
    """Test DLQ persistence and error recording."""

    def test_record_and_load_failure(self, tmp_path: Path) -> None:
        dlq_file = tmp_path / "failed.json"
        dlq = DeadLetterQueue(filepath=dlq_file)

        assert dlq.load_failures() == []

        entry = dlq.record_failure(
            url="https://example.com/dead",
            source="hackernews",
            error=requests.Timeout("Connection timed out"),
            details={"retry_count": 1},
        )

        assert entry["url"] == "https://example.com/dead"
        assert entry["source"] == "hackernews"
        assert entry["error_type"] == "transient"
        assert "timestamp" in entry

        failures = dlq.load_failures()
        assert len(failures) == 1
        assert failures[0]["url"] == "https://example.com/dead"

    def test_clear_failures(self, tmp_path: Path) -> None:
        dlq_file = tmp_path / "failed.json"
        dlq = DeadLetterQueue(filepath=dlq_file)

        dlq.record_failure("https://a.com", "rss", "Error A")
        dlq.record_failure("https://b.com", "github", "Error B")
        assert len(dlq.load_failures()) == 2

        dlq.clear()
        assert dlq.load_failures() == []


# =============================================================================
# Netlify Credit Monitoring Tests
# =============================================================================

class TestNetlifyCreditMonitoring:
    """Test credit monitoring logic."""

    def test_skip_when_no_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            status = check_netlify_credits(auth_token=None)
            assert status["status"] == "skipped"
            assert status["credits_available"] is True

    def test_success_with_valid_token(self) -> None:
        mock_sess = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_sess.get.return_value = mock_resp

        status = check_netlify_credits(auth_token="valid-token", session=mock_sess)
        assert status["status"] == "ok"
        assert status["credits_available"] is True

    def test_warning_on_unauthorized(self) -> None:
        mock_sess = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_sess.get.return_value = mock_resp

        status = check_netlify_credits(auth_token="bad-token", session=mock_sess)
        assert status["status"] == "warning"
        assert status["credits_available"] is False

    def test_exceeded_on_rate_limit(self) -> None:
        mock_sess = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_sess.get.return_value = mock_resp

        status = check_netlify_credits(auth_token="tok", session=mock_sess)
        assert status["status"] == "exceeded"
        assert status["credits_available"] is False
