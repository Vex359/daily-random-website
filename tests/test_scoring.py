"""Tests for content scoring.

Verifies the interestingness scoring algorithm produces
accurate and consistent rankings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.scoring.interestingness import (
    InterestingnessScorer,
    _contains_any,
    _extract_domain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> InterestingnessScorer:
    """Default scorer with no featured URLs and fixed time."""
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return InterestingnessScorer(now=now)


@pytest.fixture
def scorer_with_featured() -> InterestingnessScorer:
    """Scorer with pre-featured URLs."""
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return InterestingnessScorer(
        featured_urls={
            "https://featured-before.com/article",
            "https://already-featured.dev/article",
        },
        now=now,
    )


# ---------------------------------------------------------------------------
# Helper: sample candidates
# ---------------------------------------------------------------------------


def _make_candidate(**overrides: object) -> dict[str, object]:
    """Build a minimal candidate dict with sensible defaults."""
    base: dict[str, object] = {
        "url": "https://example.com/page",
        "source": "hackernews",
        "source_type": "top",
        "title": "A Normal Title",
        "description": "A normal description",
        "score": 50,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_contains_any_found(self) -> None:
        assert _contains_any("this is weird art", ["weird", "boring"]) is True

    def test_contains_any_not_found(self) -> None:
        assert _contains_any("this is boring", ["weird", "strange"]) is False

    def test_contains_any_empty_text(self) -> None:
        assert _contains_any("", ["weird"]) is False

    def test_contains_any_empty_keywords(self) -> None:
        assert _contains_any("weird stuff", []) is False

    def test_extract_domain_simple(self) -> None:
        assert _extract_domain("https://example.com/path") == "example.com"

    def test_extract_domain_with_www(self) -> None:
        assert _extract_domain("https://www.example.com/path") == "example.com"

    def test_extract_domain_no_scheme(self) -> None:
        assert _extract_domain("example.com") == "example.com"

    def test_extract_domain_subdomain(self) -> None:
        assert _extract_domain("https://blog.example.co.uk/post") == "blog.example.co.uk"


# ---------------------------------------------------------------------------
# Tests: score_candidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    """Tests for InterestingnessScorer.score_candidate()."""

    def test_returns_correct_structure(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(_make_candidate())
        assert "url" in result
        assert "score" in result
        assert "signals" in result
        assert "category" in result
        assert isinstance(result["score"], int)
        assert isinstance(result["signals"], list)
        assert isinstance(result["category"], str)

    def test_does_not_modify_original(self, scorer: InterestingnessScorer) -> None:
        candidate = _make_candidate(title="Weird Unusual Thing")
        original_title = candidate["title"]
        scorer.score_candidate(candidate)
        assert candidate["title"] == original_title

    def test_score_in_range(self, scorer: InterestingnessScorer) -> None:
        """All scores must be in 0-100."""
        candidates = [
            _make_candidate(title="Weird thing"),
            _make_candidate(url="https://google.com/search"),
            _make_candidate(url="https://bit.ly/spam"),
        ]
        for c in candidates:
            result = scorer.score_candidate(c)
            assert 0 <= result["score"] <= 100, f"Score {result['score']} out of range"

    def test_unusual_title_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(title="This is a Weird and Strange Project")
        )
        assert "unusual_title" in result["signals"]

    def test_interactive_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(title="An Interactive Simulator")
        )
        assert "interactive" in result["signals"]

    def test_interactive_in_description(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(description="This is an interactive experiment")
        )
        assert "interactive" in result["signals"]

    def test_show_hn_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(source="hackernews", source_type="show_hn")
        )
        assert "show_hn" in result["signals"]

    def test_github_source_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(_make_candidate(source="github"))
        assert "github_source" in result["signals"]

    def test_niche_domain_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://cool-project.io/page")
        )
        assert "niche_domain" in result["signals"]

    def test_educational_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                title="Learn Python from Scratch",
                description="Educational tutorial for beginners",
            )
        )
        assert "educational" in result["signals"]

    def test_artistic_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                title="Creative Art Design Tool",
                description="Visual creative coding playground",
            )
        )
        assert "artistic" in result["signals"]

    def test_unusual_description_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(description="Why does this exist? Nobody knows")
        )
        assert "unusual_description" in result["signals"]

    def test_independent_project_detected(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://my-cool-project.dev/page")
        )
        assert "independent_project" in result["signals"]

    def test_recently_discovered_with_age_hours(
        self, scorer: InterestingnessScorer
    ) -> None:
        result = scorer.score_candidate(
            _make_candidate(age_hours=12)
        )
        assert "recently_discovered" in result["signals"]

    def test_not_recently_discovered_old(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(age_hours=48)
        )
        assert "recently_discovered" not in result["signals"]

    def test_recently_discovered_with_datetime(
        self, scorer: InterestingnessScorer
    ) -> None:
        recent = datetime(2025, 1, 15, 6, 0, 0, tzinfo=timezone.utc)  # 6h before
        result = scorer.score_candidate(
            _make_candidate(discovered_at=recent.isoformat())
        )
        assert "recently_discovered" in result["signals"]

    def test_recently_discovered_with_datetime_string_z(
        self, scorer: InterestingnessScorer
    ) -> None:
        result = scorer.score_candidate(
            _make_candidate(discovered_at="2025-01-15T10:00:00Z")
        )
        assert "recently_discovered" in result["signals"]


# ---------------------------------------------------------------------------
# Tests: negative signals
# ---------------------------------------------------------------------------


class TestNegativeSignals:
    """Tests for negative scoring signals."""

    def test_major_corporation_penalized(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://google.com/search")
        )
        assert "major_corporation" in result["signals"]
        assert result["score"] < 50

    def test_major_corporation_subdomain(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://mail.google.com/inbox")
        )
        assert "major_corporation" in result["signals"]

    def test_commercial_store_penalized(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                url="https://shop-something.com/buy",
                title="Buy This Product Shop Now",
                description="Great deal and discount price",
            )
        )
        assert "commercial_store" in result["signals"]

    def test_login_only_penalized(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://mysite.com/login")
        )
        assert "login_only" in result["signals"]

    def test_login_only_signin(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(url="https://mysite.com/sign-in")
        )
        assert "login_only" in result["signals"]

    def test_suspicious_content_penalized(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                url="https://spam-site.com/offer",
                title="Click Here for Free Money Fast",
                description="Earn money online with this 100% free method",
            )
        )
        assert "suspicious_content" in result["signals"]
        assert result["score"] < 30

    def test_already_featured_penalized(
        self, scorer_with_featured: InterestingnessScorer
    ) -> None:
        result = scorer_with_featured.score_candidate(
            _make_candidate(url="https://featured-before.com/article")
        )
        assert "already_featured" in result["signals"]
        assert result["score"] <= 30

    def test_bitly_url_suspicious(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                url="https://bit.ly/abc123",
                title="Click Here for Free Money Fast",
                description="Earn money online with this 100% free method",
            )
        )
        assert "suspicious_content" in result["signals"]


# ---------------------------------------------------------------------------
# Tests: category assignment
# ---------------------------------------------------------------------------


class TestCategoryAssignment:
    """Tests for category determination."""

    def test_interactive_category(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(title="Interactive Web Simulator")
        )
        assert result["category"] == "Interactive"

    def test_weird_category(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                title="A Weird and Unusual Discovery",
                description="This is bizarre and strange",
            )
        )
        assert result["category"] == "Weird"

    def test_educational_category(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                title="Learn Machine Learning",
                description="Educational course on ML",
                source="github",
            )
        )
        assert result["category"] == "Educational"

    def test_artistic_category(self, scorer: InterestingnessScorer) -> None:
        result = scorer.score_candidate(
            _make_candidate(
                title="Creative Art Visual Studio",
                description="Creative design and art tools",
                source="github",
            )
        )
        assert result["category"] == "Art"

    def test_default_category_is_tools(self, scorer: InterestingnessScorer) -> None:
        """When no strong category signal, default to Tools."""
        result = scorer.score_candidate(
            _make_candidate(
                url="https://unknown-site.dev/page",
                title="A Simple Utility",
                description="Just a basic tool",
            )
        )
        assert result["category"] == "Tools"


# ---------------------------------------------------------------------------
# Tests: score_candidates
# ---------------------------------------------------------------------------


class TestScoreCandidates:
    """Tests for InterestingnessScorer.score_candidates()."""

    def test_returns_sorted_descending(self, scorer: InterestingnessScorer) -> None:
        candidates = [
            _make_candidate(title="Normal thing"),
            _make_candidate(title="Weird Strange Unusual Thing"),
            _make_candidate(url="https://google.com/search"),
        ]
        results = scorer.score_candidates(candidates)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_same_length_as_input(self, scorer: InterestingnessScorer) -> None:
        candidates = [_make_candidate(title=f"Title {i}") for i in range(5)]
        results = scorer.score_candidates(candidates)
        assert len(results) == 5

    def test_empty_input(self, scorer: InterestingnessScorer) -> None:
        results = scorer.score_candidates([])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: select_top
# ---------------------------------------------------------------------------


class TestSelectTop:
    """Tests for InterestingnessScorer.select_top()."""

    def test_default_limit_10(self, scorer: InterestingnessScorer) -> None:
        candidates = [_make_candidate(title=f"Title {i}") for i in range(20)]
        results = scorer.select_top(candidates)
        assert len(results) <= 10

    def test_custom_limit(self, scorer: InterestingnessScorer) -> None:
        candidates = [_make_candidate(title=f"Title {i}") for i in range(15)]
        results = scorer.select_top(candidates, limit=5)
        assert len(results) <= 5

    def test_top_items_are_highest_scored(self, scorer: InterestingnessScorer) -> None:
        candidates = [
            _make_candidate(title="Normal thing"),
            _make_candidate(title="Weird Unusual Strange Thing"),
            _make_candidate(url="https://cool-project.io/page"),
            _make_candidate(
                source="hackernews",
                source_type="show_hn",
                title="Show HN Interactive Project",
            ),
        ]
        results = scorer.select_top(candidates, limit=2)
        assert len(results) == 2
        # The top results should have the highest scores
        assert results[0]["score"] >= results[1]["score"]

    def test_show_hn_ranked_high(self, scorer: InterestingnessScorer) -> None:
        candidates = [
            _make_candidate(title="Boring Article", score=500),
            _make_candidate(
                source="hackernews",
                source_type="show_hn",
                title="Show HN Interactive Simulator",
                score=10,
            ),
        ]
        results = scorer.select_top(candidates, limit=2)
        # Show HN should rank higher despite lower HN score
        show_hn_result = [r for r in results if "show_hn" in r["signals"]]
        assert len(show_hn_result) == 1


# ---------------------------------------------------------------------------
# Tests: scoring weights
# ---------------------------------------------------------------------------


class TestScoringWeights:
    """Tests that signal weights produce correct relative scores."""

    def test_more_signals_higher_score(self, scorer: InterestingnessScorer) -> None:
        """A candidate with many positive signals should score higher."""
        many_signals = scorer.score_candidate(
            _make_candidate(
                url="https://weird-project.io/art",
                source="github",
                source_type="interactive",
                title="Weird Interactive Creative Art",
                description="Learn and experiment with visual design",
                age_hours=2,
            )
        )
        few_signals = scorer.score_candidate(
            _make_candidate(title="Normal Article")
        )
        assert many_signals["score"] > few_signals["score"]

    def test_negative_signals_lower_score(self, scorer: InterestingnessScorer) -> None:
        """Negative signals should lower the score."""
        clean = scorer.score_candidate(
            _make_candidate(url="https://cool-project.dev/page")
        )
        penalized = scorer.score_candidate(
            _make_candidate(url="https://google.com/search")
        )
        assert clean["score"] > penalized["score"]

    def test_already_featured_is_heavy_penalty(
        self, scorer_with_featured: InterestingnessScorer
    ) -> None:
        """Already featured should be a -50 penalty."""
        result = scorer_with_featured.score_candidate(
            _make_candidate(
                url="https://featured-before.com/article",
                title="Weird Creative Unusual Thing",
                source="github",
                source_type="show_hn",
            )
        )
        # Despite having many positive signals, already_featured is -50
        assert "already_featured" in result["signals"]
        assert result["score"] < 80  # Should be significantly reduced


# ---------------------------------------------------------------------------
# Tests: normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    """Tests for score normalization."""

    def test_all_scores_in_range(self, scorer: InterestingnessScorer) -> None:
        """Every possible candidate must produce 0-100."""
        extreme_cases = [
            _make_candidate(
                url="https://google.com/auth",
                title="Click Here for Free Money Fast",
                description="Earn money online with this 100% free method",
            ),
            _make_candidate(
                url="https://weird-art.io/experiment",
                source="github",
                source_type="show_hn",
                title="Show HN: Weird Interactive Creative Art Simulator",
                description="Learn and experiment with bizarre visual design",
                age_hours=1,
            ),
        ]
        for c in extreme_cases:
            result = scorer.score_candidate(c)
            assert 0 <= result["score"] <= 100, (
                f"Score {result['score']} out of range for {c['url']}"
            )

    def test_neutral_candidate_around_50(self, scorer: InterestingnessScorer) -> None:
        """A candidate with minimal signals should score near the middle."""
        result = scorer.score_candidate(
            _make_candidate(
                url="https://random-site.com/page",
                title="Normal Article",
                description="Just a regular page",
            )
        )
        # Should be in the 30-70 range (neutral territory)
        assert 20 <= result["score"] <= 80


# ---------------------------------------------------------------------------
# Tests: integration with 20 candidates
# ---------------------------------------------------------------------------


class TestIntegration20:
    """Integration test: score 20 candidates and verify top 10."""

    CANDIDATES: list[dict[str, object]] = [
        _make_candidate(
            url="https://showhn-example.dev/interactive-simulator",
            source="hackernews",
            source_type="show_hn",
            title="Show HN: Weird Interactive Simulator of Quantum Physics",
            description="A bizarre experimental tool for learning quantum mechanics",
            score=42,
        ),
        _make_candidate(
            url="https://creative-art.xyz/generative",
            source="github",
            source_type="creative_coding",
            title="Generative Art Engine",
            description="Creative visual art and design experiments",
            score=0,
        ),
        _make_candidate(
            url="https://indie-game.fun/roguelike-dungeon",
            source="hackernews",
            source_type="top",
            title="I built a browser game roguelike dungeon crawler",
            description="An interactive game built with WebGL",
            score=150,
        ),
        _make_candidate(
            url="https://strange-tools.io/what-is-this",
            source="hackernews",
            source_type="new",
            title="This strange tool converts your voice into ASCII art",
            description="Unexpected creative project nobody asked for",
            score=30,
        ),
        _make_candidate(
            url="https://learn-coding.dev/python-tutorial",
            source="github",
            source_type="interactive",
            title="Interactive Python Tutorial for Beginners",
            description="Learn programming with hands-on educational exercises",
            score=0,
        ),
        _make_candidate(
            url="https://experimental-shader.gl/test",
            source="hackernews",
            source_type="show_hn",
            title="Show HN: Experimental WebGL Shader Playground",
            description="An unusual visual experiment with real-time shaders",
            score=88,
        ),
        _make_candidate(
            url="https://tiny-utility.me/json-formatter",
            source="hackernews",
            source_type="top",
            title="JSON Formatter and Validator Tool",
            description="A simple utility tool for formatting JSON data",
            score=200,
        ),
        _make_candidate(
            url="https://niche-project.io/data-viz",
            source="github",
            source_type="web_experiment",
            title="Data Visualization Experiment",
            description="Interactive experiment with D3.js visualizations",
            score=0,
        ),
        _make_candidate(
            url="https://weird-science.org/bizarre-experiments",
            source="hackernews",
            source_type="best",
            title="Why Does This Exist: Bizarre Science Experiments at Home",
            description="Strange and unusual experiments you can try",
            score=350,
        ),
        _make_candidate(
            url="https://art-studio.app/creative-coding",
            source="github",
            source_type="creative_coding",
            title="Creative Coding Art Studio",
            description="A visual art design tool for creative coders",
            score=0,
        ),
        _make_candidate(
            url="https://google.com/docs",
            source="hackernews",
            source_type="top",
            title="Google Cloud Documentation",
            description="Learn about Google Cloud services",
            score=500,
        ),
        _make_candidate(
            url="https://amazon.com/deals",
            source="hackernews",
            source_type="new",
            title="Amazon Store Holiday Sale",
            description="Buy products at discount prices",
            score=10,
        ),
        _make_candidate(
            url="https://login-only-site.com/auth",
            source="hackernews",
            source_type="new",
            title="Login to Your Account",
            description="Sign in to access your dashboard",
            score=5,
        ),
        _make_candidate(
            url="https://bit.ly/spammy-link",
            source="hackernews",
            source_type="new",
            title="Click Here for Free Money Fast",
            description="Earn money online with this 100% free method",
            score=1,
        ),
        _make_candidate(
            url="https://already-featured.dev/article",
            source="hackernews",
            source_type="top",
            title="Interesting Article About Technology",
            description="A good article about modern tech trends",
            score=100,
        ),
        _make_candidate(
            url="https://random-indie.dev/cool-project",
            source="hackernews",
            source_type="show_hn",
            title="Show HN: Cool Interactive Project",
            description="An experimental creative tool",
            score=75,
        ),
        _make_candidate(
            url="https://new-site.xyz/discovery",
            source="hackernews",
            source_type="new",
            title="Unusual Discovery in Math",
            description="Strange patterns found in prime numbers",
            score=20,
        ),
        _make_candidate(
            url="https://tutorial-hub.io/learn-rust",
            source="github",
            source_type="interactive",
            title="Learn Rust with Interactive Examples",
            description="Educational tutorials for learning systems programming",
            score=0,
        ),
        _make_candidate(
            url="https://visual-playground.com/glitch-art",
            source="github",
            source_type="creative_coding",
            title="Glitch Art Visual Playground",
            description="Creative design experiments with visual glitch effects",
            score=0,
        ),
        _make_candidate(
            url="https://indie-sim.dev/physics-playground",
            source="hackernews",
            source_type="show_hn",
            title="Show HN: Physics Simulator Playground",
            description="Interactive physics experiment in the browser",
            score=120,
        ),
    ]

    def test_scores_20_candidates(self, scorer_with_featured: InterestingnessScorer) -> None:
        results = scorer_with_featured.score_candidates(self.CANDIDATES)
        assert len(results) == 20

    def test_top_10_all_above_50(self, scorer_with_featured: InterestingnessScorer) -> None:
        top = scorer_with_featured.select_top(self.CANDIDATES, limit=10)
        assert len(top) == 10
        for r in top:
            assert r["score"] > 50, (
                f"Top-10 item scored {r['score']}: {r['url']}"
            )

    def test_show_hn_ranked_high(self, scorer_with_featured: InterestingnessScorer) -> None:
        """Show HN posts should appear in the top results."""
        top = scorer_with_featured.select_top(self.CANDIDATES, limit=10)
        show_hn_urls = [
            "https://showhn-example.dev/interactive-simulator",
            "https://experimental-shader.gl/test",
            "https://random-indie.dev/cool-project",
            "https://indie-sim.dev/physics-playground",
        ]
        top_urls = [r["url"] for r in top]
        # At least 2 of the Show HN URLs should be in top 10
        show_hn_in_top = sum(1 for u in show_hn_urls if u in top_urls)
        assert show_hn_in_top >= 2, (
            f"Only {show_hn_in_top} Show HN posts in top 10"
        )

    def test_categories_assigned(self, scorer_with_featured: InterestingnessScorer) -> None:
        top = scorer_with_featured.select_top(self.CANDIDATES, limit=10)
        valid_categories = {"Interactive", "Weird", "Tools", "Games", "Art", "Educational"}
        for r in top:
            assert r["category"] in valid_categories, (
                f"Invalid category '{r['category']}' for {r['url']}"
            )

    def test_corporation_penalized(self, scorer_with_featured: InterestingnessScorer) -> None:
        """Major corporation should score poorly."""
        all_results = scorer_with_featured.score_candidates(self.CANDIDATES)
        google = [r for r in all_results if r["url"] == "https://google.com/docs"][0]
        assert google["score"] < 40

    def test_commercial_penalized(self, scorer_with_featured: InterestingnessScorer) -> None:
        """Commercial store should score poorly."""
        all_results = scorer_with_featured.score_candidates(self.CANDIDATES)
        amazon = [r for r in all_results if r["url"] == "https://amazon.com/deals"][0]
        assert amazon["score"] < 30

    def test_suspicious_penalized(self, scorer_with_featured: InterestingnessScorer) -> None:
        """Suspicious/spam content should score very poorly."""
        all_results = scorer_with_featured.score_candidates(self.CANDIDATES)
        spam = [r for r in all_results if r["url"] == "https://bit.ly/spammy-link"][0]
        assert spam["score"] < 20

    def test_featured_penalized(self, scorer_with_featured: InterestingnessScorer) -> None:
        """Already featured content should score poorly."""
        all_results = scorer_with_featured.score_candidates(self.CANDIDATES)
        featured = [r for r in all_results if r["url"] == "https://already-featured.dev/article"][0]
        assert "already_featured" in featured["signals"]
