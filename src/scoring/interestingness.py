"""Content scoring and ranking.

Scores collected content by interestingness to determine
which posts to feature on the daily website.

Usage as script:
    python -m src.scoring.interestingness
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal weights
# ---------------------------------------------------------------------------

# Positive signals
POSITIVE_SIGNALS: dict[str, dict[str, Any]] = {
    "unusual_title": {
        "weight": 20,
        "keywords": ["weird", "strange", "unusual", "experimental"],
        "field": "title",
    },
    "interactive": {
        "weight": 15,
        "keywords": ["interactive", "simulator", "experiment", "game"],
        "field": "title_and_description",
    },
    "independent_project": {
        "weight": 15,
    },
    "show_hn": {
        "weight": 15,
    },
    "recently_discovered": {
        "weight": 10,
    },
    "niche_domain": {
        "weight": 10,
    },
    "unusual_description": {
        "weight": 10,
        "keywords": ["why does this exist", "bizarre", "unexpected"],
        "field": "description",
    },
    "github_source": {
        "weight": 10,
    },
    "educational": {
        "weight": 5,
        "keywords": ["learn", "tutorial", "educational"],
        "field": "title_and_description",
    },
    "artistic": {
        "weight": 5,
        "keywords": ["art", "creative", "design", "visual"],
        "field": "title_and_description",
    },
}

# Negative signals
NEGATIVE_SIGNALS: dict[str, dict[str, Any]] = {
    "major_corporation": {"weight": -30},
    "commercial_store": {"weight": -30},
    "login_only": {"weight": -30},
    "suspicious_content": {"weight": -50},
    "already_featured": {"weight": -50},
}

# ---------------------------------------------------------------------------
# Keyword lists for detection
# ---------------------------------------------------------------------------

_MAJOR_CORPORATIONS: set[str] = {
    "google.com", "apple.com", "microsoft.com", "amazon.com", "facebook.com",
    "meta.com", "twitter.com", "x.com", "linkedin.com", "netflix.com",
    "youtube.com", "github.com", "stackoverflow.com", "reddit.com",
    "wikipedia.org", "yahoo.com", "bing.com", "cloudflare.com",
    "salesforce.com", "oracle.com", "ibm.com", "intel.com", "amd.com",
    "nvidia.com", "samsung.com", "sony.com", "adobe.com", "vmware.com",
    "shopify.com", "stripe.com", "slack.com", "dropbox.com", "zoom.us",
    "twitch.tv", "pinterest.com", "snapchat.com", "tiktok.com",
    "bytedance.com", "tesla.com", "spacex.com", "openai.com",
}

_COMMERCIAL_KEYWORDS: set[str] = {
    "shop", "store", "buy", "cart", "checkout", "purchase", "price",
    "deal", "discount", "coupon", "sale", "offer", "order", "pay",
    "subscribe", "pricing", "plans", "enterprise",
}

_UTILTIY_KEYWORDS: set[str] = {
    "tool", "utility", "converter", "calculator", "generator",
    "formatter", "validator", "checker", "analyzer", "editor",
    "builder", "maker", "creator", "helper", "assistant",
    "manager", "organizer", "planner", "scheduler", "tracker",
}

_GAME_KEYWORDS: set[str] = {
    "game", "play", "arcade", "puzzle", "rpg", "roguelike",
    "platformer", "tetris", "chess", "strategy", "browser game",
    "html5 game", "web game", "indie game",
}

# ---------------------------------------------------------------------------
# Category mapping: signal -> category label
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, str] = {
    "interactive": "Interactive",
    "unusual_title": "Weird",
    "unusual_description": "Weird",
    "niche_domain": "Tools",
    "github_source": "Tools",
    "show_hn": "Interactive",  # Show HN defaults to Interactive unless overridden
    "educational": "Educational",
    "artistic": "Art",
}

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class InterestingnessScorer:
    """Scores website candidates by interestingness using a weighted signal system.

    No network requests, no AI — pure keyword and heuristic scoring.
    """

    def __init__(
        self,
        *,
        featured_urls: set[str] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Initialise the scorer.

        Args:
            featured_urls: URLs already featured in previous runs. Candidates
                matching these receive a heavy penalty.
            now: Current timestamp for recency checks. Defaults to UTC now.
        """
        self._featured_urls: set[str] = featured_urls or set()
        self._now = now or datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Score a single candidate.

        Returns a dict with keys: url, score, signals, category.
        The original candidate is never modified.
        """
        signals: list[str] = []
        raw_score = 0.0

        # --- Negative signals (checked first so they always apply) ---
        neg = self._check_negative_signals(candidate)
        raw_score += neg["score"]
        signals.extend(neg["signals"])

        # --- Positive signals ---
        pos = self._check_positive_signals(candidate)
        raw_score += pos["score"]
        signals.extend(pos["signals"])

        # --- Normalize to 0-100 ---
        normalized = self._normalize(raw_score)

        # --- Determine category from highest-weight positive signal ---
        category = self._determine_category(pos["top_signal"], signals)

        return {
            "url": candidate.get("url", ""),
            "score": normalized,
            "signals": signals,
            "category": category,
        }

    def score_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score and rank multiple candidates by interestingness.

        Returns a list sorted by score descending (highest first).
        """
        scored = [self.score_candidate(c) for c in candidates]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def select_top(
        self, candidates: list[dict[str, Any]], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Score candidates and return the top N by interestingness."""
        ranked = self.score_candidates(candidates)
        return ranked[:limit]

    # ------------------------------------------------------------------
    # Internal: signal detection
    # ------------------------------------------------------------------

    def _check_positive_signals(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect all positive signals and return cumulative score."""
        score = 0.0
        signals: list[str] = []
        top_signal: str | None = None
        top_weight = 0

        title = (candidate.get("title") or "").lower()
        description = (candidate.get("description") or "").lower()
        url = candidate.get("url", "")
        source = candidate.get("source", "")
        source_type = candidate.get("source_type", "")

        # Unusual title
        kw_cfg = POSITIVE_SIGNALS["unusual_title"]
        if _contains_any(title, kw_cfg["keywords"]):
            _add_signal("unusual_title", kw_cfg["weight"], score, signals)
            score += kw_cfg["weight"]
            if kw_cfg["weight"] > top_weight:
                top_signal = "unusual_title"
                top_weight = kw_cfg["weight"]

        # Interactive keywords
        kw_cfg = POSITIVE_SIGNALS["interactive"]
        combined = f"{title} {description}"
        if _contains_any(combined, kw_cfg["keywords"]):
            score += kw_cfg["weight"]
            signals.append("interactive")
            if kw_cfg["weight"] > top_weight:
                top_signal = "interactive"
                top_weight = kw_cfg["weight"]

        # Independent project (not a major corporation)
        if self._is_independent(url):
            score += POSITIVE_SIGNALS["independent_project"]["weight"]
            signals.append("independent_project")
            w = POSITIVE_SIGNALS["independent_project"]["weight"]
            if w > top_weight:
                top_signal = "independent_project"
                top_weight = w

        # Show HN source
        if source == "hackernews" and source_type == "show_hn":
            score += POSITIVE_SIGNALS["show_hn"]["weight"]
            signals.append("show_hn")
            w = POSITIVE_SIGNALS["show_hn"]["weight"]
            if w > top_weight:
                top_signal = "show_hn"
                top_weight = w

        # Recently discovered (within 24h)
        if self._is_recent(candidate):
            score += POSITIVE_SIGNALS["recently_discovered"]["weight"]
            signals.append("recently_discovered")
            w = POSITIVE_SIGNALS["recently_discovered"]["weight"]
            if w > top_weight:
                top_signal = "recently_discovered"
                top_weight = w

        # Niche domain
        if self._is_niche(url):
            score += POSITIVE_SIGNALS["niche_domain"]["weight"]
            signals.append("niche_domain")
            w = POSITIVE_SIGNALS["niche_domain"]["weight"]
            if w > top_weight:
                top_signal = "niche_domain"
                top_weight = w

        # Unusual description
        kw_cfg = POSITIVE_SIGNALS["unusual_description"]
        if _contains_any(description, kw_cfg["keywords"]):
            score += kw_cfg["weight"]
            signals.append("unusual_description")
            if kw_cfg["weight"] > top_weight:
                top_signal = "unusual_description"
                top_weight = kw_cfg["weight"]

        # GitHub source
        if source == "github":
            score += POSITIVE_SIGNALS["github_source"]["weight"]
            signals.append("github_source")
            w = POSITIVE_SIGNALS["github_source"]["weight"]
            if w > top_weight:
                top_signal = "github_source"
                top_weight = w

        # Educational keywords
        kw_cfg = POSITIVE_SIGNALS["educational"]
        if _contains_any(combined, kw_cfg["keywords"]):
            score += kw_cfg["weight"]
            signals.append("educational")
            if kw_cfg["weight"] > top_weight:
                top_signal = "educational"
                top_weight = kw_cfg["weight"]

        # Artistic keywords
        kw_cfg = POSITIVE_SIGNALS["artistic"]
        if _contains_any(combined, kw_cfg["keywords"]):
            score += kw_cfg["weight"]
            signals.append("artistic")
            if kw_cfg["weight"] > top_weight:
                top_signal = "artistic"
                top_weight = kw_cfg["weight"]

        return {"score": score, "signals": signals, "top_signal": top_signal}

    def _check_negative_signals(
        self, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect negative signals and return cumulative penalty."""
        score = 0.0
        signals: list[str] = []

        url = candidate.get("url", "")
        title = (candidate.get("title") or "").lower()
        description = (candidate.get("description") or "").lower()

        # Major corporation
        if self._is_major_corporation(url):
            score += NEGATIVE_SIGNALS["major_corporation"]["weight"]
            signals.append("major_corporation")

        # Commercial store
        combined = f"{title} {description}"
        if _contains_any(combined, _COMMERCIAL_KEYWORDS):
            # Only flag if URL also looks commercial or title is very commercial
            if self._looks_commercial(url, title):
                score += NEGATIVE_SIGNALS["commercial_store"]["weight"]
                signals.append("commercial_store")

        # Login-only page
        if self._is_login_only(url):
            score += NEGATIVE_SIGNALS["login_only"]["weight"]
            signals.append("login_only")

        # Suspicious content
        if self._is_suspicious(candidate):
            score += NEGATIVE_SIGNALS["suspicious_content"]["weight"]
            signals.append("suspicious_content")

        # Already featured
        if url in self._featured_urls:
            score += NEGATIVE_SIGNALS["already_featured"]["weight"]
            signals.append("already_featured")

        return {"score": score, "signals": signals}

    # ------------------------------------------------------------------
    # Internal: signal helpers
    # ------------------------------------------------------------------

    def _is_independent(self, url: str) -> bool:
        """Return True if the URL is from an independent project (not a major corp)."""
        return not self._is_major_corporation(url)

    def _is_major_corporation(self, url: str) -> bool:
        """Return True if the URL belongs to a major corporation."""
        domain = _extract_domain(url)
        # Check exact match and one level of subdomain
        if domain in _MAJOR_CORPORATIONS:
            return True
        # Check parent domain for subdomains like mail.google.com
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            return parent in _MAJOR_CORPORATIONS
        return False

    def _is_niche(self, url: str) -> bool:
        """Return True if the domain is niche (not mainstream)."""
        domain = _extract_domain(url)
        # Mainstream domains that are well-known but not corporations
        mainstream = {
            "wikipedia.org", "github.com", "stackoverflow.com",
            "medium.com", "dev.to", "hackernews.com", "news.ycombinator.com",
            "reddit.com", "twitter.com", "x.com", "youtube.com",
            "linkedin.com", "facebook.com", "instagram.com",
        }
        if domain in mainstream or self._is_major_corporation(url):
            return False
        # Check if it's a .dev, .io, .xyz, .me TLD — often indie
        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        indie_tlds = {"dev", "io", "xyz", "me", "app", "sh", "fm", "tech", "fun"}
        if tld in indie_tlds:
            return True
        # Unknown domain → niche
        return True

    def _is_recent(self, candidate: dict[str, Any]) -> bool:
        """Return True if the candidate was discovered within the last 24 hours."""
        age_hours = candidate.get("age_hours")
        if age_hours is not None:
            return float(age_hours) < 24

        discovered_at = candidate.get("discovered_at")
        if discovered_at is not None:
            try:
                if isinstance(discovered_at, str):
                    dt = datetime.fromisoformat(discovered_at.replace("Z", "+00:00"))
                elif isinstance(discovered_at, datetime):
                    dt = discovered_at
                else:
                    return False
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                delta = self._now - dt
                return delta.total_seconds() < 86400  # 24 hours
            except (ValueError, TypeError):
                return False
        return False

    def _is_login_only(self, url: str) -> bool:
        """Return True if the URL looks like a login-only page."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        login_paths = {
            "/login", "/signin", "/sign-in", "/auth", "/authenticate",
            "/sso", "/oauth", "/accounts/login", "/account/signin",
        }
        if path.rstrip("/") in login_paths:
            return True
        # Query param based login
        query = parsed.query.lower()
        if "redirect" in query and ("login" in query or "signin" in query):
            return True
        return False

    def _looks_commercial(self, url: str, title: str) -> bool:
        """Return True if the URL and title look like a commercial store."""
        domain = _extract_domain(url)
        # Check if domain itself contains commercial keywords
        for kw in _COMMERCIAL_KEYWORDS:
            if kw in domain:
                return True
        # Check if title is heavily commercial (2+ commercial keywords)
        commercial_count = sum(1 for kw in _COMMERCIAL_KEYWORDS if kw in title)
        return commercial_count >= 2

    def _is_suspicious(self, candidate: dict[str, Any]) -> bool:
        """Return True if the content looks suspicious/spammy."""
        title = (candidate.get("title") or "").lower()
        description = (candidate.get("description") or "").lower()
        url = candidate.get("url", "")
        combined = f"{title} {description}"

        # Spam indicators
        spam_patterns = [
            "click here", "free money", "earn money fast",
            "make money online", "get rich quick", "limited time offer",
            "act now", "congratulations you won", "claim your prize",
            "100% free", "no credit card", "weight loss",
            "buy followers", "cheap followers",
        ]
        spam_count = sum(1 for p in spam_patterns if p in combined)
        if spam_count >= 2:
            return True

        # Suspicious URL patterns
        suspicious_url_patterns = [
            "bit.ly/", "tinyurl.com/", "t.co/",
            "goo.gl/", "ow.ly/", "is.gd/",
        ]
        for pattern in suspicious_url_patterns:
            if pattern in url.lower():
                return True

        return False

    # ------------------------------------------------------------------
    # Normalization and categorization
    # ------------------------------------------------------------------

    def _normalize(self, raw_score: float) -> int:
        """Normalize raw score to 0-100 range.

        Uses a practical range of -50 to +80 for mapping.
        """
        practical_min = -50.0
        practical_max = 80.0
        normalized = (raw_score - practical_min) / (practical_max - practical_min) * 100
        return max(0, min(100, round(normalized)))

    def _determine_category(
        self, top_signal: str | None, all_signals: list[str] | None = None
    ) -> str:
        """Determine the content category from the detected signals.

        Prefers content-type signals (interactive, unusual, educational, artistic)
        over source-type signals (github_source, show_hn, independent_project)
        for category assignment.
        """
        # Content-type signals that directly indicate category
        content_signals = ["interactive", "unusual_title", "unusual_description",
                           "educational", "artistic"]
        # Prefer content signals over source signals
        signals_to_check = (all_signals or [])
        for sig in signals_to_check:
            if sig in content_signals and sig in _CATEGORY_MAP:
                return _CATEGORY_MAP[sig]
        # Fall back to top signal
        if top_signal and top_signal in _CATEGORY_MAP:
            return _CATEGORY_MAP[top_signal]
        return "Tools"  # Default category


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return True if text contains any of the keywords."""
    return any(kw in text for kw in keywords)


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _add_signal(
    signal: str, weight: float, current_score: float, signals: list[str]
) -> None:
    """Append a signal name to the signals list."""
    signals.append(signal)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# 20 test candidates covering all signal types
_TEST_CANDIDATES: list[dict[str, Any]] = [
    {
        "url": "https://showhn-example.dev/interactive-simulator",
        "source": "hackernews",
        "source_type": "show_hn",
        "title": "Show HN: Weird Interactive Simulator of Quantum Physics",
        "description": "A bizarre experimental tool for learning quantum mechanics",
        "score": 42,
    },
    {
        "url": "https://creative-art.xyz/generative",
        "source": "github",
        "source_type": "creative_coding",
        "title": "Generative Art Engine",
        "description": "Creative visual art and design experiments",
        "score": 0,
    },
    {
        "url": "https://indie-game.fun/roguelike-dungeon",
        "source": "hackernews",
        "source_type": "top",
        "title": "I built a browser game roguelike dungeon crawler",
        "description": "An interactive game built with WebGL",
        "score": 150,
    },
    {
        "url": "https://strange-tools.io/what-is-this",
        "source": "hackernews",
        "source_type": "new",
        "title": "This strange tool converts your voice into ASCII art",
        "description": "Unexpected creative project nobody asked for",
        "score": 30,
    },
    {
        "url": "https://learn-coding.dev/python-tutorial",
        "source": "github",
        "source_type": "interactive",
        "title": "Interactive Python Tutorial for Beginners",
        "description": "Learn programming with hands-on educational exercises",
        "score": 0,
    },
    {
        "url": "https://experimental-shader.gl/test",
        "source": "hackernews",
        "source_type": "show_hn",
        "title": "Show HN: Experimental WebGL Shader Playground",
        "description": "An unusual visual experiment with real-time shaders",
        "score": 88,
    },
    {
        "url": "https://tiny-utility.me/json-formatter",
        "source": "hackernews",
        "source_type": "top",
        "title": "JSON Formatter and Validator Tool",
        "description": "A simple utility tool for formatting JSON data",
        "score": 200,
    },
    {
        "url": "https://niche-project.io/data-viz",
        "source": "github",
        "source_type": "web_experiment",
        "title": "Data Visualization Experiment",
        "description": "Interactive experiment with D3.js visualizations",
        "score": 0,
    },
    {
        "url": "https://weird-science.org/bizarre-experiments",
        "source": "hackernews",
        "source_type": "best",
        "title": "Why Does This Exist: Bizarre Science Experiments at Home",
        "description": "Strange and unusual experiments you can try",
        "score": 350,
    },
    {
        "url": "https://art-studio.app/creative-coding",
        "source": "github",
        "source_type": "creative_coding",
        "title": "Creative Coding Art Studio",
        "description": "A visual art design tool for creative coders",
        "score": 0,
    },
    {
        "url": "https://google.com/docs",
        "source": "hackernews",
        "source_type": "top",
        "title": "Google Cloud Documentation",
        "description": "Learn about Google Cloud services",
        "score": 500,
    },
    {
        "url": "https://amazon.com/deals",
        "source": "hackernews",
        "source_type": "new",
        "title": "Amazon Store Holiday Sale",
        "description": "Buy products at discount prices",
        "score": 10,
    },
    {
        "url": "https://login-only-site.com/auth",
        "source": "hackernews",
        "source_type": "new",
        "title": "Login to Your Account",
        "description": "Sign in to access your dashboard",
        "score": 5,
    },
    {
        "url": "https://bit.ly/spammy-link",
        "source": "hackernews",
        "source_type": "new",
        "title": "Click Here for Free Money Fast",
        "description": "Earn money online with this 100% free method",
        "score": 1,
    },
    {
        "url": "https://already-featured.dev/article",
        "source": "hackernews",
        "source_type": "top",
        "title": "Interesting Article About Technology",
        "description": "A good article about modern tech trends",
        "score": 100,
    },
    {
        "url": "https://random-indie.dev/cool-project",
        "source": "hackernews",
        "source_type": "show_hn",
        "title": "Show HN: Cool Interactive Project",
        "description": "An experimental creative tool",
        "score": 75,
    },
    {
        "url": "https://new-site.xyz/discovery",
        "source": "hackernews",
        "source_type": "new",
        "title": "Unusual Discovery in Math",
        "description": "Strange patterns found in prime numbers",
        "score": 20,
    },
    {
        "url": "https://tutorial-hub.io/learn-rust",
        "source": "github",
        "source_type": "interactive",
        "title": "Learn Rust with Interactive Examples",
        "description": "Educational tutorials for learning systems programming",
        "score": 0,
    },
    {
        "url": "https://visual-playground.com/glitch-art",
        "source": "github",
        "source_type": "creative_coding",
        "title": "Glitch Art Visual Playground",
        "description": "Creative design experiments with visual glitch effects",
        "score": 0,
    },
    {
        "url": "https://indie-sim.dev/physics-playground",
        "source": "hackernews",
        "source_type": "show_hn",
        "title": "Show HN: Physics Simulator Playground",
        "description": "Interactive physics experiment in the browser",
        "score": 120,
    },
]


def main() -> None:
    """CLI entry point: score 20 test candidates and display results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Pre-set some URLs as already featured for the penalty test
    featured = {"https://already-featured.dev/article", "https://featured-before.com/article"}
    scorer = InterestingnessScorer(featured_urls=featured)

    print(f"\n{'='*72}")
    print("INTERESTINGNESS SCORING ALGORITHM — DEMO")
    print(f"{'='*72}")
    print(f"Scoring {len(_TEST_CANDIDATES)} candidates...\n")

    results = scorer.select_top(_TEST_CANDIDATES, limit=10)

    print(f"\n{'='*72}")
    print(f"TOP {len(results)} RESULTS")
    print(f"{'='*72}\n")

    for i, result in enumerate(results, 1):
        signals_str = ", ".join(result["signals"]) if result["signals"] else "none"
        print(f"{i:2d}. [{result['score']:3d}] [{result['category']:>12}] {result['url']}")
        print(f"    Signals: {signals_str}")
        print()

    # Summary stats
    all_scored = scorer.score_candidates(_TEST_CANDIDATES)
    high_score = [r for r in all_scored if r["score"] > 50]
    low_score = [r for r in all_scored if r["score"] <= 50]

    print(f"{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  Total candidates scored: {len(all_scored)}")
    print(f"  High score (>50):       {len(high_score)}")
    print(f"  Low score (<=50):       {len(low_score)}")
    print(f"  Top score:              {all_scored[0]['score']} ({all_scored[0]['url']})")
    print(f"  Bottom score:           {all_scored[-1]['score']} ({all_scored[-1]['url']})")

    # Validation checks
    print(f"\n{'='*72}")
    print("VALIDATION")
    print(f"{'='*72}")

    checks = [
        ("20 candidates scored", len(all_scored) == 20),
        ("Top 10 returned", len(results) == 10),
        ("Top 10 all score >50", all(r["score"] > 50 for r in results)),
        ("Show HN posts ranked high", any(
            r["url"] == "https://showhn-example.dev/interactive-simulator"
            and r["score"] > 70
            for r in results
        )),
        ("Major corp penalized", any(
            r["url"] == "https://google.com/docs" and r["score"] < 40
            for r in all_scored
        )),
        ("Commercial store penalized", any(
            r["url"] == "https://amazon.com/deals" and r["score"] < 30
            for r in all_scored
        )),
        ("Already featured penalized", any(
            "already_featured" in r["signals"]
            for r in all_scored
        )),
        ("Categories assigned", all(r["category"] for r in results)),
    ]

    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All checks passed!")
    else:
        print("Some checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
