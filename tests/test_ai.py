"""Tests for AI content generation.

Verifies that AI-generated summaries, titles, and descriptions
are produced correctly. All API calls are mocked to avoid real
network requests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from src.ai.generator import (
    FALLBACK_DESCRIPTION,
    MAX_DESCRIPTION_LENGTH,
    AIDescriptionGenerator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CANDIDATE = {
    "url": "https://example.com/cool-project",
    "title": "Cool Project",
    "description": "A very cool project that does interesting things.",
    "headings": ["About", "Features", "Download"],
    "page_text": "This is a project about doing cool things on the internet.",
}

SAMPLE_AI_RESPONSE = {
    "description": "Check out this amazing project that brings something truly unique to the web. The creativity here is off the charts! 🔥",
    "category": "Interactive",
    "why_interesting": "Innovative approach to web-based tools",
}


def _mock_success_response(content: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock httpx response with a successful API call."""
    data = content or SAMPLE_AI_RESPONSE
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(data)}}]
    }
    mock_resp.text = json.dumps(data)
    return mock_resp


def _mock_error_response(status_code: int, body: str = "") -> MagicMock:
    """Create a mock httpx response with an error status."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = body or f"Error {status_code}"
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: generate_description
# ---------------------------------------------------------------------------


class TestGenerateDescription:
    """Tests for single-candidate description generation."""

    def test_generates_valid_output(self) -> None:
        """generate_description returns proper dict with description, category, why."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert "description" in result
            assert "category" in result
            assert "why_interesting" in result
            assert isinstance(result["description"], str)
            assert len(result["description"]) > 0

    def test_api_key_from_env(self) -> None:
        """API key falls back to OPENROUTER_API_KEY environment variable."""
        with (
            patch("src.ai.generator.os.environ", {"OPENROUTER_API_KEY": "env-key"}),
            patch("src.ai.generator.httpx.Client") as mock_cls,
        ):
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator()
            assert gen._api_key == "env-key"

    def test_no_api_key_returns_fallback(self) -> None:
        """Without API key, returns fallback description."""
        with patch("src.ai.generator.os.environ", {}):
            gen = AIDescriptionGenerator(api_key="")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == FALLBACK_DESCRIPTION
            assert result["category"] == "Uncategorized"

    def test_prompt_includes_candidate_data(self) -> None:
        """User prompt includes URL, title, description, headings, and page text."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            gen.generate_description(SAMPLE_CANDIDATE)

            # Inspect the prompt sent to the API
            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            user_msg = payload["messages"][1]["content"]

            assert "https://example.com/cool-project" in user_msg
            assert "Cool Project" in user_msg
            assert "About" in user_msg
            assert "Page text:" in user_msg

    def test_page_text_limited_to_500_words(self) -> None:
        """Page text is truncated to 500 words in the prompt."""
        long_text = " ".join(["word"] * 600)
        candidate = {**SAMPLE_CANDIDATE, "page_text": long_text}

        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            gen.generate_description(candidate)

            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            user_msg = payload["messages"][1]["content"]

            # The user_msg should not contain "word word word..." 600 times
            # It should have been truncated
            word_count = len(user_msg.split("word"))
            assert word_count <= 510  # some tolerance for other text

    def test_system_prompt_matches_spec(self) -> None:
        """System prompt matches the specified social media copywriter prompt."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            gen.generate_description(SAMPLE_CANDIDATE)

            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            system_msg = payload["messages"][0]["content"]

            assert "social media copywriter" in system_msg.lower()
            assert "instagram" in system_msg.lower()
            assert "2-3 sentences" in system_msg
            assert "emojis" in system_msg.lower()

    def test_uses_free_model(self) -> None:
        """API call uses the openrouter/free model."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            gen.generate_description(SAMPLE_CANDIDATE)

            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]

            assert payload["model"] == "openrouter/free"


# ---------------------------------------------------------------------------
# Tests: generate_batch
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    """Tests for batch description generation."""

    def test_batch_returns_same_length(self) -> None:
        """generate_batch returns one result per input candidate."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_success_response()
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            candidates = [SAMPLE_CANDIDATE, SAMPLE_CANDIDATE, SAMPLE_CANDIDATE]
            results = gen.generate_batch(candidates)

            assert len(results) == 3
            for r in results:
                assert "description" in r

    def test_batch_with_empty_list(self) -> None:
        """generate_batch with empty list returns empty list."""
        gen = AIDescriptionGenerator(api_key="test-key")
        results = gen.generate_batch([])
        assert results == []

    def test_batch_continues_on_failure(self) -> None:
        """generate_batch continues generating even if one candidate fails."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_error_response(500, "Server error")
            return _mock_success_response()

        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.side_effect = side_effect
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            candidates = [SAMPLE_CANDIDATE, SAMPLE_CANDIDATE]
            results = gen.generate_batch(candidates)

            assert len(results) == 2
            # First result should be fallback (due to 500 error)
            assert results[0]["description"] == FALLBACK_DESCRIPTION
            # Second should be the AI response
            assert results[1]["description"] == SAMPLE_AI_RESPONSE["description"]


# ---------------------------------------------------------------------------
# Tests: validate_output
# ---------------------------------------------------------------------------


class TestValidateOutput:
    """Tests for output validation."""

    def test_valid_output(self) -> None:
        """Valid output passes validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        result = gen.validate_output(SAMPLE_AI_RESPONSE)

        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_missing_description_key(self) -> None:
        """Missing 'description' key fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        incomplete = {"category": "X", "why_interesting": "Y"}
        result = gen.validate_output(incomplete)

        assert result["is_valid"] is False
        assert any("Missing required key" in i and "description" in i for i in result["issues"])

    def test_missing_category_key(self) -> None:
        """Missing 'category' key fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        incomplete = {"description": "A" * 80, "why_interesting": "Y"}
        result = gen.validate_output(incomplete)

        assert result["is_valid"] is False
        assert any("category" in i for i in result["issues"])

    def test_description_too_short(self) -> None:
        """Description under 50 chars fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        short = {
            "description": "Too short.",
            "category": "X",
            "why_interesting": "Y",
        }
        result = gen.validate_output(short)

        assert result["is_valid"] is False
        assert any("too short" in i.lower() for i in result["issues"])

    def test_description_too_long(self) -> None:
        """Description over 300 chars fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        long_desc = {
            "description": "X" * 301,
            "category": "X",
            "why_interesting": "Y",
        }
        result = gen.validate_output(long_desc)

        assert result["is_valid"] is False
        assert any("too long" in i.lower() for i in result["issues"])

    def test_empty_category(self) -> None:
        """Empty category string fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        no_cat = {
            "description": "A" * 80,
            "category": "",
            "why_interesting": "Y",
        }
        result = gen.validate_output(no_cat)

        assert result["is_valid"] is False
        assert any("category" in i.lower() and "empty" in i.lower() for i in result["issues"])

    def test_empty_why_interesting(self) -> None:
        """Empty why_interesting fails validation."""
        gen = AIDescriptionGenerator(api_key="test-key")
        no_why = {
            "description": "A" * 80,
            "category": "X",
            "why_interesting": "",
        }
        result = gen.validate_output(no_why)

        assert result["is_valid"] is False
        assert any("why_interesting" in i for i in result["issues"])

    def test_boundary_length_50(self) -> None:
        """Description at exactly 50 chars passes."""
        gen = AIDescriptionGenerator(api_key="test-key")
        boundary = {
            "description": "A" * 50,
            "category": "X",
            "why_interesting": "Y",
        }
        result = gen.validate_output(boundary)
        assert result["is_valid"] is True

    def test_boundary_length_300(self) -> None:
        """Description at exactly 300 chars passes."""
        gen = AIDescriptionGenerator(api_key="test-key")
        boundary = {
            "description": "A" * 300,
            "category": "X",
            "why_interesting": "Y",
        }
        result = gen.validate_output(boundary)
        assert result["is_valid"] is True


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for API error handling and fallback behavior."""

    def test_http_500_returns_fallback(self) -> None:
        """HTTP 500 error returns fallback description."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            client.post.return_value = _mock_error_response(500)
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == FALLBACK_DESCRIPTION

    def test_invalid_json_returns_fallback(self) -> None:
        """Invalid JSON in response returns fallback description."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "not valid json at all"}}]
            }
            mock_resp.text = "not valid json"
            client.post.return_value = mock_resp
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == FALLBACK_DESCRIPTION

    def test_empty_response_returns_fallback(self) -> None:
        """Empty API response returns fallback description."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": ""}}]
            }
            mock_resp.text = ""
            client.post.return_value = mock_resp
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == FALLBACK_DESCRIPTION

    def test_malformed_response_structure_returns_fallback(self) -> None:
        """Missing 'choices' key returns fallback description."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"error": "bad request"}
            mock_resp.text = '{"error": "bad request"}'
            client.post.return_value = mock_resp
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == FALLBACK_DESCRIPTION

    def test_json_in_code_block_is_parsed(self) -> None:
        """Response wrapped in ```json code block is parsed correctly."""
        with patch("src.ai.generator.httpx.Client") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            content = "```json\n" + json.dumps(SAMPLE_AI_RESPONSE) + "\n```"
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": content}}]
            }
            mock_resp.text = content
            client.post.return_value = mock_resp
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)

            gen = AIDescriptionGenerator(api_key="test-key")
            result = gen.generate_description(SAMPLE_CANDIDATE)

            assert result["description"] == SAMPLE_AI_RESPONSE["description"]
            assert result["category"] == SAMPLE_AI_RESPONSE["category"]


# ---------------------------------------------------------------------------
# Tests: rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_enforcement(self) -> None:
        """Rate limiter tracks request timestamps."""
        gen = AIDescriptionGenerator(api_key="test-key")
        gen._rate_limit_times = []

        # Simulate 20 requests within the window
        for _ in range(20):
            gen._check_rate_limit()

        assert len(gen._rate_limit_times) == 20

    def test_rate_limit_prunes_old_entries(self) -> None:
        """Rate limiter prunes timestamps outside the window."""
        gen = AIDescriptionGenerator(api_key="test-key")
        import time

        # Add 20 timestamps that are all old (61 seconds ago)
        old_time = time.monotonic() - 61.0
        gen._rate_limit_times = [old_time] * 20

        # This should not sleep — old entries should be pruned
        gen._check_rate_limit()

        # Should have 21 entries (20 old pruned + 1 new)
        # Actually old entries are pruned, so only the new one
        assert len(gen._rate_limit_times) <= 2


# ---------------------------------------------------------------------------
# Tests: context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_enter_exit(self) -> None:
        """AIDescriptionGenerator works as a context manager."""
        with AIDescriptionGenerator(api_key="test-key") as gen:
            assert isinstance(gen, AIDescriptionGenerator)

    def test_close_closes_session(self) -> None:
        """close() closes the HTTP session."""
        gen = AIDescriptionGenerator(api_key="test-key")
        gen._session = MagicMock()
        gen.close()
        gen._session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_prompt_format(self) -> None:
        """User prompt contains all expected sections."""
        gen = AIDescriptionGenerator(api_key="test-key")
        prompt = gen._build_user_prompt(SAMPLE_CANDIDATE)

        assert "URL:" in prompt
        assert "Title:" in prompt
        assert "Description:" in prompt
        assert "Headings:" in prompt
        assert "Page text:" in prompt
        assert "JSON" in prompt

    def test_prompt_without_optional_fields(self) -> None:
        """Prompt handles missing optional fields gracefully."""
        gen = AIDescriptionGenerator(api_key="test-key")
        minimal = {"url": "https://test.com", "title": "Test"}
        prompt = gen._build_user_prompt(minimal)

        assert "URL: https://test.com" in prompt
        assert "Title: Test" in prompt
        assert "Headings:" not in prompt
        assert "Page text:" not in prompt

    def test_prompt_with_many_headings(self) -> None:
        """Prompt limits headings to first 10."""
        gen = AIDescriptionGenerator(api_key="test-key")
        candidate = {
            **SAMPLE_CANDIDATE,
            "headings": [f"Heading {i}" for i in range(15)],
        }
        prompt = gen._build_user_prompt(candidate)

        assert "Heading 9" in prompt
        assert "Heading 10" not in prompt  # 0-indexed, should be limited


# ---------------------------------------------------------------------------
# Tests: _parse_ai_response edge cases
# ---------------------------------------------------------------------------


class TestParseAIResponse:
    """Tests for AI response parsing edge cases."""

    def test_parse_plain_json(self) -> None:
        """Plain JSON string is parsed correctly."""
        gen = AIDescriptionGenerator(api_key="test-key")
        raw = json.dumps(SAMPLE_AI_RESPONSE)
        result = gen._parse_ai_response(raw, SAMPLE_CANDIDATE)

        assert result["description"] == SAMPLE_AI_RESPONSE["description"]

    def test_parse_json_in_code_block(self) -> None:
        """JSON inside markdown code block is extracted."""
        gen = AIDescriptionGenerator(api_key="test-key")
        raw = "```json\n" + json.dumps(SAMPLE_AI_RESPONSE) + "\n```"
        result = gen._parse_ai_response(raw, SAMPLE_CANDIDATE)

        assert result["description"] == SAMPLE_AI_RESPONSE["description"]

    def test_parse_json_in_bare_code_block(self) -> None:
        """JSON inside bare code block (no language tag) is extracted."""
        gen = AIDescriptionGenerator(api_key="test-key")
        raw = "```\n" + json.dumps(SAMPLE_AI_RESPONSE) + "\n```"
        result = gen._parse_ai_response(raw, SAMPLE_CANDIDATE)

        assert result["description"] == SAMPLE_AI_RESPONSE["description"]

    def test_parse_json_with_surrounding_text(self) -> None:
        """JSON embedded in surrounding text is extracted."""
        gen = AIDescriptionGenerator(api_key="test-key")
        prefix = "Here is the result:\n"
        suffix = "\n\nDone."
        raw = prefix + json.dumps(SAMPLE_AI_RESPONSE) + suffix
        result = gen._parse_ai_response(raw, SAMPLE_CANDIDATE)

        assert result["description"] == SAMPLE_AI_RESPONSE["description"]

    def test_parse_completely_invalid_returns_fallback(self) -> None:
        """Completely invalid text returns fallback."""
        gen = AIDescriptionGenerator(api_key="test-key")
        result = gen._parse_ai_response("hello world nothing here", SAMPLE_CANDIDATE)

        assert result["description"] == FALLBACK_DESCRIPTION

    def test_parse_missing_keys_get_defaults(self) -> None:
        """Response missing keys get default values."""
        gen = AIDescriptionGenerator(api_key="test-key")
        incomplete = {"description": "A" * 80}  # missing category and why
        result = gen._parse_ai_response(json.dumps(incomplete), SAMPLE_CANDIDATE)

        assert result["category"] == "Uncategorized"
        assert result["why_interesting"] == "Curated for your interest"


# ---------------------------------------------------------------------------
# Tests: fix_description
# ---------------------------------------------------------------------------


class TestFixDescription:
    """Tests for description auto-fix."""

    def test_fix_trims_long_description(self) -> None:
        """Long description is trimmed to max length."""
        gen = AIDescriptionGenerator(api_key="test-key")
        long_desc = {
            "description": "A" * 400,
            "category": "X",
            "why_interesting": "Y",
        }
        fixed = gen._fix_description(long_desc)
        assert len(fixed["description"]) <= MAX_DESCRIPTION_LENGTH

    def test_fix_trims_at_sentence_boundary(self) -> None:
        """Long description is trimmed at sentence boundary when possible."""
        gen = AIDescriptionGenerator(api_key="test-key")
        # Build a description that's over 300 chars but has sentence breaks
        desc = "This is sentence one. " + "X" * 200 + " Final sentence."
        long_desc = {
            "description": desc,
            "category": "X",
            "why_interesting": "Y",
        }
        fixed = gen._fix_description(long_desc)
        assert len(fixed["description"]) <= MAX_DESCRIPTION_LENGTH

