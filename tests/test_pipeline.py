"""Unit and integration tests for ContentPipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline import ContentPipeline
from src.resilience import DeadLetterQueue
from src.storage.manager import StorageManager

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def temp_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    return data_dir, screenshots_dir, dist_dir


@pytest.fixture()
def mock_storage(temp_dirs: tuple[Path, Path, Path]) -> StorageManager:
    data_dir, _, _ = temp_dirs
    return StorageManager(
        posts_path=data_dir / "posts.json",
        seen_domains_path=data_dir / "seen_domains.json",
    )


from unittest.mock import MagicMock


@pytest.fixture()
def pipeline(temp_dirs: tuple[Path, Path, Path], mock_storage: StorageManager) -> ContentPipeline:
    data_dir, screenshots_dir, dist_dir = temp_dirs
    dlq = DeadLetterQueue(filepath=data_dir / "failed.json")
    return ContentPipeline(
        storage_manager=mock_storage,
        ai_generator=MagicMock(),
        screenshot_capture=MagicMock(),
        dlq=dlq,
        screenshots_dir=screenshots_dir,
        output_dir=dist_dir,
    )


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelineCollectors:
    """Test collector invocation and error handling."""

    def test_collectors_registration(self, pipeline: ContentPipeline) -> None:
        collectors = pipeline.get_collectors()
        assert "hackernews" in collectors
        assert "github" in collectors
        assert "rss" in collectors
        assert "wikipedia" in collectors
        assert "awesome_lists" in collectors

    def test_single_source_collection(self, pipeline: ContentPipeline) -> None:
        with patch.object(
            pipeline,
            "get_collectors",
            return_value={
                "rss": lambda limit: [{"url": "https://indie.example.com", "title": "Indie", "source": "rss"}],
                "hackernews": lambda limit: [{"url": "https://hn.example.com", "title": "HN", "source": "hackernews"}],
            },
        ):
            results = pipeline.collect(source="rss", limit_per_source=10)
            assert len(results) == 1
            assert results[0]["source"] == "rss"

    def test_invalid_source_raises_error(self, pipeline: ContentPipeline) -> None:
        with pytest.raises(ValueError, match="Unknown source"):
            pipeline.collect(source="invalid_source")

    def test_fallback_when_one_source_fails(self, pipeline: ContentPipeline) -> None:
        """If Hacker News fails, the pipeline continues with remaining sources."""
        def failing_hn(limit: int):
            raise ConnectionError("HN API is down")

        def working_rss(limit: int):
            return [{"url": "https://rss.example.com", "title": "RSS Pick", "source": "rss"}]

        with patch.object(
            pipeline,
            "get_collectors",
            return_value={"hackernews": failing_hn, "rss": working_rss},
        ):
            candidates = pipeline.collect()
            assert len(candidates) == 1
            assert candidates[0]["url"] == "https://rss.example.com"

            # Check that DLQ recorded the HN failure
            failures = pipeline.dlq.load_failures()
            assert len(failures) == 1
            assert failures[0]["source"] == "hackernews"


class TestNormalizationAndDedup:
    """Test URL normalization and domain deduplication in pipeline."""

    def test_dedup_and_domain_filtering(self, pipeline: ContentPipeline) -> None:
        # Pre-seed seen domain in storage
        pipeline.storage.add_post({
            "url": "https://alreadyseen.com",
            "domain": "alreadyseen.com",
            "title": "Old",
            "category": "Interactive",
            "discovered_at": "2026-01-01T00:00:00Z",
        })

        candidates = [
            {"url": "https://alreadyseen.com/page", "title": "Seen"},
            {"url": "https://fresh.example.com", "title": "Fresh"},
            {"url": "https://fresh.example.com/sub", "title": "Fresh Sub (Duplicate Domain)"},
            {"url": "https://google.com", "title": "Corp"},
        ]

        deduped = pipeline.normalize_and_deduplicate(candidates)
        assert len(deduped) == 1
        assert deduped[0]["domain"] == "example.com"


class TestScoringAndRanking:
    """Test candidate scoring and ranking."""

    def test_scoring_order(self, pipeline: ContentPipeline) -> None:
        candidates = [
            {"url": "https://boring.com", "title": "Ordinary site", "description": ""},
            {"url": "https://cool.com", "title": "Weird interactive simulation game", "description": "unexpected tool"},
        ]

        scored = pipeline.score_candidates(candidates)
        assert len(scored) == 2
        assert scored[0]["domain"] == "cool.com"
        assert scored[0]["score"] > scored[1]["score"]


class TestEnrichmentAndFallbacks:
    """Test screenshot and AI description generation with fallbacks."""

    def test_enrich_with_screenshot_fallback(self, pipeline: ContentPipeline) -> None:
        candidate = {
            "url": "https://cool-exp.org",
            "domain": "cool-exp.org",
            "title": "Cool Experiment",
            "category": "Interactive",
            "score": 88,
        }

        # Simulate screenshot failure
        with patch.object(
            pipeline.screenshot,
            "capture_screenshot",
            side_effect=TimeoutError("Screenshot timed out"),
        ):
            post = pipeline.enrich_candidate(candidate, dry_run=False)
            assert post["url"] == "https://cool-exp.org"
            assert "screenshots/cool-exp-org.webp" in post["screenshot_path"]
            # Placeholder file was generated
            img_path = pipeline.screenshots_dir / "cool-exp-org.webp"
            assert img_path.exists()

    def test_enrich_with_ai_fallback(self, pipeline: ContentPipeline) -> None:
        candidate = {
            "url": "https://cool-exp.org",
            "domain": "cool-exp.org",
            "title": "Cool Experiment",
            "category": "Interactive",
            "score": 88,
        }

        # Simulate AI generator failure
        with patch.object(
            pipeline.ai,
            "generate_description",
            side_effect=Exception("OpenRouter 429"),
        ):
            post = pipeline.enrich_candidate(candidate, dry_run=False)
            assert post["ai_description"] is not None
            assert len(post["ai_description"]) > 0


class TestPipelineEndToEnd:
    """Test full pipeline execution."""

    def test_full_run_happy_path(self, pipeline: ContentPipeline) -> None:
        mock_candidates = [
            {
                "url": "https://neal.fun",
                "title": "Neal Fun Interactive Games",
                "description": "Strange interactive web toys",
                "source": "hackernews",
            }
        ]

        with (
            patch.object(pipeline, "collect", return_value=mock_candidates),
            patch.object(pipeline, "validate_and_inspect", return_value=mock_candidates),
            patch.object(pipeline.screenshot, "capture_screenshot"),
            patch.object(
                pipeline.ai,
                "generate_description",
                return_value={
                    "description": "An incredible playground of internet oddities! 🚀",
                    "category": "Games",
                    "why_interesting": "Pure fun experiments",
                },
            ),
        ):
            res = pipeline.run(dry_run=False, limit=1)
            assert res["success"] is True
            assert len(res["published_posts"]) == 1
            assert res["site_generated"] is True

            posts = pipeline.storage.load_posts()
            assert len(posts) == 1
            assert posts[0]["domain"] == "neal.fun"

            # Verify dist/index.html was generated
            assert (pipeline.output_dir / "index.html").exists()

    def test_dry_run_does_not_mutate_storage(self, pipeline: ContentPipeline) -> None:
        mock_candidates = [
            {"url": "https://dryrun.org", "title": "Dry Run Site", "source": "rss"}
        ]

        with (
            patch.object(pipeline, "collect", return_value=mock_candidates),
            patch.object(pipeline, "validate_and_inspect", return_value=mock_candidates),
        ):
            res = pipeline.run(dry_run=True, limit=1)
            assert res["success"] is True
            assert res["dry_run"] is True

            # Storage should remain empty
            assert pipeline.storage.load_posts() == []
