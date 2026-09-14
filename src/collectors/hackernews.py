"""Hacker News API content collector.

Fetches URLs from the official Hacker News Firebase API
(https://hacker-news.firebaseio.com/v0/). Respects rate limits
(1 req/sec) and handles errors gracefully.

Usage as script:
    python -m src.collectors.hackernews
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://hacker-news.firebaseio.com/v0"

# Endpoints mapping story type -> API path
STORY_ENDPOINTS: dict[str, str] = {
    "new": "/newstories.json",
    "top": "/topstories.json",
    "best": "/beststories.json",
}


class HackerNewsCollectorError(Exception):
    """Raised when the collector encounters an unrecoverable error."""


class HackerNewsCollector:
    """Collects URLs from the Hacker News API.

    Rate-limited to 1 request per second. All public methods return a list
    of dicts with keys: url, source, source_type, title, score.
    """

    def __init__(self, *, timeout: int = 30, rate_limit_delay: float = 1.0) -> None:
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "DailyRandomWebsite/1.0"})

    # -- Internal helpers -----------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, path: str) -> Any:
        """GET a JSON resource from the HN API with rate limiting and error handling."""
        self._rate_limit()
        url = f"{BASE_URL}{path}"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s", url)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
            return None
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching %s: %s", url, exc)
            return None
        except ValueError:
            logger.warning("Invalid JSON from %s", url)
            return None

    @staticmethod
    def _extract_url_from_item(item: dict[str, Any], source_type: str) -> dict[str, Any] | None:
        """Extract a result dict from a HN story item, or None if no URL."""
        url = item.get("url")
        if not url:
            # Ask HN, Poll, etc. may have no external URL
            return None
        return {
            "url": url,
            "source": "hackernews",
            "source_type": source_type,
            "title": item.get("title", ""),
            "score": item.get("score", 0),
        }

    def _fetch_story_ids(self, endpoint_key: str) -> list[int]:
        """Fetch a list of story IDs for a given endpoint."""
        path = STORY_ENDPOINTS.get(endpoint_key)
        if not path:
            logger.error("Unknown endpoint key: %s", endpoint_key)
            return []
        data = self._get_json(path)
        if data is None or not isinstance(data, list):
            return []
        return [int(sid) for sid in data if isinstance(sid, (int, float))]

    def _fetch_items(self, story_ids: list[int], limit: int) -> list[dict[str, Any]]:
        """Fetch item details for a list of story IDs, up to *limit*."""
        results: list[dict[str, Any]] = []
        for sid in story_ids:
            if len(results) >= limit:
                break
            item = self._get_json(f"/item/{sid}.json")
            if item is None or not isinstance(item, dict):
                continue
            # Only stories with external URLs
            if item.get("url"):
                results.append(item)
        return results

    # -- Public API ------------------------------------------------------------

    def fetch_new_stories(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch new stories from HN.

        Returns list of URL dicts (max *limit* items, only stories with URLs).
        """
        story_ids = self._fetch_story_ids("new")
        if not story_ids:
            logger.warning("No new story IDs returned from HN API")
            return []
        items = self._fetch_items(story_ids[:limit], limit)
        return [self._extract_url_from_item(item, "new") for item in items
                if self._extract_url_from_item(item, "new") is not None]

    def fetch_top_stories(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch top stories from HN."""
        story_ids = self._fetch_story_ids("top")
        if not story_ids:
            logger.warning("No top story IDs returned from HN API")
            return []
        items = self._fetch_items(story_ids[:limit], limit)
        return [self._extract_url_from_item(item, "top") for item in items
                if self._extract_url_from_item(item, "top") is not None]

    def fetch_best_stories(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch best stories from HN."""
        story_ids = self._fetch_story_ids("best")
        if not story_ids:
            logger.warning("No best story IDs returned from HN API")
            return []
        items = self._fetch_items(story_ids[:limit], limit)
        return [self._extract_url_from_item(item, "best") for item in items
                if self._extract_url_from_item(item, "best") is not None]

    def fetch_show_hn(self, limit: int = 50, scan_limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Show HN posts.

        Show HN posts are among the most valuable content on HN because
        they represent original projects and contributions. We pull from
        top stories and filter for title starting with 'Show HN:'.
        scan_limit caps how many story IDs to scan (avoids iterating all 500).
        """
        story_ids = self._fetch_story_ids("top")
        if not story_ids:
            return []

        results: list[dict[str, Any]] = []
        for sid in story_ids[:scan_limit]:
            if len(results) >= limit:
                break
            item = self._get_json(f"/item/{sid}.json")
            if item is None or not isinstance(item, dict):
                continue
            title = item.get("title", "")
            if title.lower().startswith("show hn:"):
                extracted = self._extract_url_from_item(item, "show_hn")
                if extracted is not None:
                    results.append(extracted)
        return results

    def collect_all(self, limit: int = 200) -> list[dict[str, Any]]:
        """Combine all HN sources into a single deduplicated list of URLs.

        Sources are combined in priority order: show_hn, best, top, new.
        Up to *limit* total results (default 200).
        """
        collected: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        # Priority order: Show HN first, then best, top, new
        source_configs = [
            ("show_hn", self.fetch_show_hn, 50),
            ("best", self.fetch_best_stories, 80),
            ("top", self.fetch_top_stories, 80),
            ("new", self.fetch_new_stories, 80),
        ]

        for source_name, fetcher, batch_limit in source_configs:
            if len(collected) >= limit:
                break
            remaining = limit - len(collected)
            effective_limit = min(batch_limit, remaining)
            if source_name == "show_hn":
                batch = fetcher(limit=effective_limit, scan_limit=min(effective_limit * 3, 150))
            else:
                batch = fetcher(limit=effective_limit)
            for item in batch:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    collected.append(item)

        logger.info(
            "Collected %d unique URLs from HN (limit=%d)", len(collected), limit
        )
        return collected[:limit]

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> HackerNewsCollector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def main() -> None:
    """CLI entry point: fetch HN URLs and print them."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    with HackerNewsCollector() as collector:
        results = collector.collect_all(limit=200)
        print(f"\n{'='*60}")
        print(f"Collected {len(results)} URLs from Hacker News")
        print(f"{'='*60}\n")
        for i, item in enumerate(results, 1):
            print(f"{i:3d}. [{item['source_type']:>7}] {item['title'][:70]}")
            print(f"     {item['url']}")
            print()
    if len(results) >= 10:
        print(f"\n✓ PASS: {len(results)} URLs collected (≥10 required)")
    else:
        print(f"\n✗ FAIL: Only {len(results)} URLs collected (≥10 required)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
