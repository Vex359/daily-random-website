"""Pipeline orchestrator for the Daily Random Website.

Coordinates the full autonomous content pipeline:
1. Collect content candidates from multiple sources (HN, GitHub, RSS, Wikipedia, Awesome Lists)
2. Filter, normalize, and deduplicate
3. Validate HTML and safety
4. Score by interestingness
5. Enhance top candidate with AI-generated descriptions
6. Capture Playwright screenshot
7. Persist to git-based JSON storage (data/posts.json)
8. Generate the static site (dist/)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.ai.generator import FALLBACK_DESCRIPTION, AIDescriptionGenerator
from src.collectors.awesome_lists import AwesomeListsCollector
from src.collectors.github_collector import GitHubCollector
from src.collectors.hackernews import HackerNewsCollector
from src.collectors.rss import RSSCollector
from src.collectors.wikipedia import WikipediaCollector
from src.filters.dedup import URLNormalizer
from src.filters.domains import DomainFilter
from src.filters.quality import QualityFilter
from src.filters.safety import SafetyFilter
from src.resilience import DeadLetterQueue, classify_error
from src.scoring.interestingness import InterestingnessScorer
from src.screenshot.capture import (
    ScreenshotCapture,
    _create_placeholder,
    url_to_filename,
)
from src.site.generator import SiteGenerator
from src.storage.manager import StorageManager

logger = logging.getLogger(__name__)

# Base project paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_SCREENSHOTS_DIR = _PROJECT_ROOT / "screenshots"
_DIST_DIR = _PROJECT_ROOT / "dist"


class ContentPipeline:
    """Orchestrates candidate collection, filtering, scoring, enrichment, and publishing."""

    def __init__(
        self,
        *,
        storage_manager: StorageManager | None = None,
        site_generator: SiteGenerator | None = None,
        scorer: InterestingnessScorer | None = None,
        ai_generator: AIDescriptionGenerator | None = None,
        screenshot_capture: ScreenshotCapture | None = None,
        normalizer: URLNormalizer | None = None,
        domain_filter: DomainFilter | None = None,
        safety_filter: SafetyFilter | None = None,
        quality_filter: QualityFilter | None = None,
        dlq: DeadLetterQueue | None = None,
        screenshots_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        request_timeout: float = 8.0,
    ) -> None:
        self.storage = storage_manager or StorageManager()
        self.site_generator = site_generator or SiteGenerator(output_dir=output_dir or _DIST_DIR)
        self.scorer = scorer or InterestingnessScorer(featured_urls=self.storage.get_seen_domains())
        self._ai = ai_generator
        self._screenshot = screenshot_capture
        self.normalizer = normalizer or URLNormalizer(seen_domains_path=self.storage._seen_domains_path)
        self.domain_filter = domain_filter or DomainFilter()
        self.safety_filter = safety_filter or SafetyFilter(domain_filter=self.domain_filter)
        self.quality_filter = quality_filter or QualityFilter()
        self.dlq = dlq or DeadLetterQueue()
        self.screenshots_dir = Path(screenshots_dir) if screenshots_dir else _SCREENSHOTS_DIR
        self.output_dir = Path(output_dir) if output_dir else _DIST_DIR
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DailyRandomWebsiteBot/1.0 (+https://github.com/daily-random-website)"
        })

    @property
    def ai(self) -> AIDescriptionGenerator:
        """Lazy-loaded AI description generator."""
        if self._ai is None:
            self._ai = AIDescriptionGenerator()
        return self._ai

    @property
    def screenshot(self) -> ScreenshotCapture:
        """Lazy-loaded screenshot capture instance."""
        if self._screenshot is None:
            self._screenshot = ScreenshotCapture()
        return self._screenshot

    # =========================================================================
    # Step 1: Collection with Source Fallbacks
    # =========================================================================

    def get_collectors(self) -> dict[str, Callable[[int], list[dict[str, Any]]]]:
        """Return available collector factory functions."""
        def _collect_rss(limit: int) -> list[dict[str, Any]]:
            col = RSSCollector()
            col.load_curated_feeds()
            return col.collect_all(limit=limit)

        return {
            "hackernews": lambda limit: HackerNewsCollector().collect_all(limit=limit),
            "github": lambda limit: GitHubCollector().collect_all(limit=limit),
            "rss": _collect_rss,
            "wikipedia": lambda limit: WikipediaCollector().collect_all(limit=limit),
            "awesome_lists": lambda limit: AwesomeListsCollector().collect_all(limit=limit),
        }

    def collect(
        self,
        source: str | None = None,
        limit_per_source: int = 40,
        simulate_hn_failure: bool = False,
    ) -> list[dict[str, Any]]:
        """Collect candidates from sources with per-source fault tolerance.

        Args:
            source: Specific source name or None for all sources.
            limit_per_source: Max items per source.
            simulate_hn_failure: Flag to simulate primary source failure.

        Returns:
            Aggregated candidate list from all successful sources.
        """
        all_collectors = self.get_collectors()
        if source and source != "all":
            if source not in all_collectors:
                raise ValueError(f"Unknown source: '{source}'. Valid sources: {list(all_collectors.keys())}")
            active_collectors = {source: all_collectors[source]}
        else:
            active_collectors = all_collectors

        collected: list[dict[str, Any]] = []

        for src_name, fetcher in active_collectors.items():
            try:
                if simulate_hn_failure and src_name == "hackernews":
                    raise ConnectionError("Simulated Hacker News API connection timeout")

                logger.info("Collecting from source: %s (limit=%d)", src_name, limit_per_source)
                items = fetcher(limit_per_source)
                logger.info("Source %s returned %d items", src_name, len(items))
                for item in items:
                    if "source" not in item:
                        item["source"] = src_name
                    collected.append(item)
            except Exception as exc:  # noqa: BLE001
                err_type = classify_error(exc)
                logger.warning(
                    "Source %s failed with %s error: %s. Continuing with remaining sources.",
                    src_name,
                    err_type,
                    exc,
                )
                self.dlq.record_failure(
                    url=f"source://{src_name}",
                    source=src_name,
                    error=exc,
                    error_type=err_type,
                    details={"phase": "collection"},
                )

        logger.info("Total candidates collected across all active sources: %d", len(collected))
        return collected

    # =========================================================================
    # Step 2: Normalization & Deduplication
    # =========================================================================

    def normalize_and_deduplicate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize URLs, extract domains, and remove duplicate / seen domains."""
        seen_domains = self.storage.get_seen_domains()
        unique_candidates: list[dict[str, Any]] = []
        seen_in_batch: set[str] = set()

        for c in candidates:
            raw_url = c.get("url", "").strip()
            if not raw_url:
                continue

            try:
                canonical = self.normalizer.normalize_url(raw_url)
                domain = self.normalizer.extract_domain(canonical)
            except (ValueError, TypeError, KeyError) as exc:
                logger.debug("Failed to normalize URL %s: %s", raw_url, exc)
                continue

            if not domain:
                continue

            # Check if domain was previously featured or seen in this batch
            if domain in seen_domains or domain in seen_in_batch:
                continue

            # Check domain blocklists
            if self.domain_filter.is_blocked(domain) or self.domain_filter.is_major_corporation(domain):
                continue

            seen_in_batch.add(domain)
            candidate_copy = dict(c)
            candidate_copy["url"] = raw_url
            candidate_copy["canonical_url"] = canonical
            candidate_copy["domain"] = domain
            unique_candidates.append(candidate_copy)

        logger.info("Deduplication: %d candidates remaining from %d", len(unique_candidates), len(candidates))
        return unique_candidates

    # =========================================================================
    # Step 3: Validation & Content Inspection
    # =========================================================================

    def validate_and_inspect(
        self,
        candidates: list[dict[str, Any]],
        max_to_inspect: int = 25,
    ) -> list[dict[str, Any]]:
        """Perform HTTP checks and extract page metadata for candidates."""
        validated: list[dict[str, Any]] = []

        for candidate in candidates[:max_to_inspect]:
            url = candidate["url"]

            # Quick safety filter check on URL string
            if not self.safety_filter.is_safe_url(url):
                continue

            # Attempt lightweight HTTP fetch
            try:
                resp = self.session.get(
                    url,
                    timeout=self.request_timeout,
                    allow_redirects=True,
                    stream=True,
                )
                safety_check = self.safety_filter.check_response(resp)
                if not safety_check["is_safe"]:
                    logger.debug("Safety check failed for %s: %s", url, safety_check["issues"])
                    continue

                # Read up to 256KB to avoid huge downloads
                raw_bytes = resp.raw.read(256 * 1024, decode_content=True)
                html_text = raw_bytes.decode(resp.encoding or "utf-8", errors="ignore")

                # Parse headings and text
                soup = BeautifulSoup(html_text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else candidate.get("title", "")
                headings = [h.get_text().strip() for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text().strip()][:5]
                page_text = " ".join(soup.stripped_strings)[:1500]

                # Run quality validation
                quality = self.quality_filter.validate_page(url, html=html_text, title=title)
                if not quality["is_valid"]:
                    logger.debug("Quality check failed for %s: %s", url, quality["issues"])
                    continue

                candidate_updated = dict(candidate)
                candidate_updated["title"] = title or candidate.get("title", "Untitled")
                candidate_updated["headings"] = headings
                candidate_updated["page_text"] = page_text
                if not candidate_updated.get("description"):
                    meta_desc = soup.find("meta", attrs={"name": "description"})
                    if meta_desc and meta_desc.get("content"):
                        candidate_updated["description"] = meta_desc["content"].strip()

                validated.append(candidate_updated)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Validation fetch failed for %s: %s", url, exc)
                # If network fails during inspection, we can keep the candidate if it has a title
                if candidate.get("title"):
                    validated.append(candidate)

        logger.info("Validation complete: %d valid candidates", len(validated))
        return validated

    # =========================================================================
    # Step 4: Scoring & Ranking
    # =========================================================================

    def score_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score candidates using the interestingness scoring algorithm."""
        scored: list[dict[str, Any]] = []
        for c in candidates:
            score_res = self.scorer.score_candidate(c)
            merged = dict(c)
            if "domain" not in merged and merged.get("url"):
                try:
                    merged["domain"] = self.normalizer.extract_domain(merged["url"])
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not extract domain from %s: %s", merged.get("url"), exc)
            merged["score"] = score_res["score"]
            merged["category"] = score_res["category"]
            merged["signals"] = score_res["signals"]
            scored.append(merged)

        # Sort descending by score
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        return scored

    # =========================================================================
    # Step 5: Content Enrichment (Screenshot + AI Description)
    # =========================================================================

    def enrich_candidate(self, candidate: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        """Capture screenshot and generate AI description for a winner candidate."""
        url = candidate["url"]
        domain = candidate.get("domain")
        if not domain:
            try:
                domain = self.normalizer.extract_domain(url)
            except Exception:  # noqa: BLE001
                domain = url

        # 1. Screenshot Capture
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        filename = url_to_filename(url)
        screenshot_path = self.screenshots_dir / filename
        rel_screenshot_path = f"screenshots/{filename}"

        if not dry_run:
            try:
                self.screenshot.capture_screenshot(url, screenshot_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Screenshot capture failed for %s: %s. Using placeholder.", url, exc)
                _create_placeholder(screenshot_path)
        else:
            rel_screenshot_path = f"screenshots/{filename}"

        # 2. AI Description Generation
        ai_desc = FALLBACK_DESCRIPTION
        why_interesting = "An intriguing discovery from the web."
        category = candidate.get("category", "Interactive")

        if not dry_run:
            try:
                ai_result = self.ai.generate_description(candidate)
                ai_desc = ai_result.get("description", FALLBACK_DESCRIPTION)
                if ai_result.get("category"):
                    category = ai_result["category"]
                if ai_result.get("why_interesting"):
                    why_interesting = ai_result["why_interesting"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("AI description generation failed for %s: %s. Using fallback.", url, exc)
        else:
            ai_desc = (
                f"[DRY-RUN] {candidate.get('title', domain)} is an interesting find discovered via "
                f"{candidate.get('source', 'curation')}. Check out this interactive experience!"
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "id": str(uuid.uuid4()),
            "url": url,
            "canonical_url": candidate.get("canonical_url", url),
            "domain": domain,
            "title": candidate.get("title") or domain,
            "description": candidate.get("description") or ai_desc,
            "ai_description": ai_desc,
            "category": category,
            "source": candidate.get("source", "hackernews"),
            "source_type": candidate.get("source_type", ""),
            "score": int(candidate.get("score", 50)),
            "screenshot_path": rel_screenshot_path,
            "why_interesting": why_interesting,
            "discovered_at": now_iso,
            "published_at": now_iso,
        }

    # =========================================================================
    # Step 6: Storage Persistence
    # =========================================================================

    def publish_post(self, post: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        """Save post to storage and enforce retention policy."""
        if dry_run:
            logger.info("[DRY-RUN] Post would be saved: %s (%s)", post["title"], post["domain"])
            return post

        added = self.storage.add_post(post)
        archived = self.storage.cleanup_old_posts(keep_count=50)
        if archived:
            logger.info("Cleaned up %d old posts under retention policy", len(archived))
        return added

    # =========================================================================
    # Step 7: Static Site Generation & Asset Sync
    # =========================================================================

    def sync_screenshot_assets(self, output_dir: Path) -> None:
        """Mirror screenshot images from screenshots/ to dist/screenshots/."""
        dist_screenshots = output_dir / "screenshots"
        dist_screenshots.mkdir(parents=True, exist_ok=True)

        if self.screenshots_dir.exists():
            for img in self.screenshots_dir.glob("*.webp"):
                dest = dist_screenshots / img.name
                if not dest.exists() or img.stat().st_mtime > dest.stat().st_mtime:
                    try:
                        shutil.copy2(img, dest)
                    except OSError as exc:
                        logger.warning("Could not sync screenshot %s: %s", img, exc)

        # Ensure placeholder exists
        placeholder_file = dist_screenshots / "placeholder.webp"
        if not placeholder_file.exists():
            _create_placeholder(placeholder_file)

    def generate_site(self, output_dir: Path | str | None = None) -> Path:
        """Generate static site from storage posts and synchronize assets."""
        out = Path(output_dir) if output_dir else self.output_dir
        posts = self.storage.load_posts()
        generated_path = self.site_generator.generate_site(posts, output_dir=out)
        self.sync_screenshot_assets(out)
        logger.info("Site successfully generated at: %s (%d posts)", generated_path, len(posts))
        return generated_path

    # =========================================================================
    # Step 8: Full Pipeline Execution
    # =========================================================================

    def run(
        self,
        *,
        dry_run: bool = False,
        source: str | None = None,
        limit: int = 1,
        simulate_hn_failure: bool = False,
    ) -> dict[str, Any]:
        """Execute the entire pipeline end-to-end.

        Args:
            dry_run: If True, do not mutate storage or call live paid services.
            source: Filter to single source or None for all.
            limit: Number of new posts to discover and publish (default 1).
            simulate_hn_failure: Flag for testing fallback behavior.

        Returns:
            Dict summary of pipeline execution.
        """
        logger.info(
            "Starting Daily Random Website pipeline (dry_run=%s, source=%s, limit=%d)",
            dry_run,
            source,
            limit,
        )

        errors: list[str] = []
        published_posts: list[dict[str, Any]] = []

        # 1. Collect
        candidates = self.collect(
            source=source,
            simulate_hn_failure=simulate_hn_failure,
        )
        if not candidates:
            logger.warning("No candidates collected from any source.")
            # Still generate site if posts exist in storage
            self.generate_site()
            return {
                "success": False,
                "dry_run": dry_run,
                "collected_count": 0,
                "filtered_count": 0,
                "scored_count": 0,
                "published_posts": [],
                "site_generated": True,
                "errors": ["No candidates collected."],
            }

        # 2. Normalize and Deduplicate
        deduped = self.normalize_and_deduplicate(candidates)

        # 3. Validate
        validated = self.validate_and_inspect(deduped)
        eval_pool = validated if validated else deduped

        # 4. Score
        scored = self.score_candidates(eval_pool)

        # 5. Select Winners and Enrich
        winners = scored[:limit]
        for winner in winners:
            try:
                post = self.enrich_candidate(winner, dry_run=dry_run)
                published = self.publish_post(post, dry_run=dry_run)
                published_posts.append(published)
                logger.info("Successfully published post: %s (%s)", published["title"], published["url"])
            except Exception as exc:
                err_msg = f"Failed to publish {winner.get('url')}: {exc}"
                logger.exception(err_msg)
                errors.append(err_msg)
                self.dlq.record_failure(
                    url=winner.get("url", "unknown"),
                    source=winner.get("source", "unknown"),
                    error=exc,
                    error_type="permanent",
                    details={"phase": "enrich_and_publish"},
                )

        # 6. Generate Site
        site_path = self.generate_site()

        return {
            "success": len(published_posts) > 0 or dry_run,
            "dry_run": dry_run,
            "collected_count": len(candidates),
            "filtered_count": len(deduped),
            "scored_count": len(scored),
            "published_posts": published_posts,
            "site_generated": site_path.exists(),
            "errors": errors,
        }
