"""Unit tests for main CLI entry point."""

from __future__ import annotations

from unittest.mock import patch

from src.main import main, parse_args


class TestCLIParsing:
    """Test argument parsing and CLI flags."""

    def test_default_arguments(self) -> None:
        opts = parse_args([])
        assert opts.dry_run is False
        assert opts.source is None
        assert opts.limit == 1
        assert opts.output_dir == "dist"
        assert opts.check_credits is False
        assert opts.test_error_handling is False
        assert opts.verbose is False

    def test_custom_arguments(self) -> None:
        opts = parse_args([
            "--dry-run",
            "--source", "github",
            "--limit", "3",
            "--output-dir", "custom_dist",
            "--verbose",
        ])
        assert opts.dry_run is True
        assert opts.source == "github"
        assert opts.limit == 3
        assert opts.output_dir == "custom_dist"
        assert opts.verbose is True

    def test_test_error_handling_flag(self) -> None:
        opts = parse_args(["--test-error-handling"])
        assert opts.test_error_handling is True

    def test_check_credits_flag(self) -> None:
        opts = parse_args(["--check-credits"])
        assert opts.check_credits is True


class TestMainExecution:
    """Test main() function paths."""

    def test_main_check_credits_success(self) -> None:
        with patch("src.main.check_netlify_credits", return_value={"status": "ok", "credits_available": True, "message": "OK"}):
            code = main(["--check-credits"])
            assert code == 0

    def test_main_check_credits_failure(self) -> None:
        with patch("src.main.check_netlify_credits", return_value={"status": "warning", "credits_available": False, "message": "Bad auth"}):
            code = main(["--check-credits"])
            assert code == 1

    def test_main_runs_pipeline_successfully(self) -> None:
        mock_result = {
            "success": True,
            "dry_run": True,
            "collected_count": 10,
            "filtered_count": 5,
            "scored_count": 5,
            "published_posts": [],
            "site_generated": True,
            "errors": [],
        }
        with patch("src.main.ContentPipeline") as mock_pipeline_cls:
            instance = mock_pipeline_cls.return_value
            instance.run.return_value = mock_result

            code = main(["--dry-run"])
            assert code == 0
            instance.run.assert_called_once_with(
                dry_run=True,
                source=None,
                limit=1,
                simulate_hn_failure=False,
            )

    def test_main_test_error_handling_passes_flag(self) -> None:
        mock_result = {
            "success": True,
            "dry_run": True,
            "collected_count": 5,
            "filtered_count": 2,
            "scored_count": 2,
            "published_posts": [],
            "site_generated": True,
            "errors": [],
        }
        with patch("src.main.ContentPipeline") as mock_pipeline_cls:
            instance = mock_pipeline_cls.return_value
            instance.run.return_value = mock_result

            code = main(["--test-error-handling", "--dry-run"])
            assert code == 0
            instance.run.assert_called_once_with(
                dry_run=True,
                source=None,
                limit=1,
                simulate_hn_failure=True,
            )
