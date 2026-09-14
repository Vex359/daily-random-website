"""RSS feed content collector.

Parses and aggregates content from curated RSS/Atom feeds
to find interesting articles and resources.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlparse, urljoin

import feedparser
import requests

logger = logging.getLogger(__name__)

# Maximum number of URLs to collect across all feeds
MAX_URLS = 100

# Seconds to wait between HTTP requests (rate limiting)
REQUEST_DELAY = 2.0

# Maximum number of redirect levels to follow
MAX_REDIRECTS = 2

# HTTP timeout in seconds
REQUEST_TIMEOUT = 15


# Curated feed list: (url, category/source_type)
CURATED_FEEDS: list[tuple[str, str]] = [
    # Indie projects
    ("https://www.indiehackers.com/feed", "indie"),
    # Web dev
    ("https://dev.to/feed", "webdev"),
    ("https://css-tricks.com/feed/", "webdev"),
    ("https://www.smashingmagazine.com/feed/", "webdev"),
    ("https://1stwebdesigner.com/feed/", "webdev"),
    # Creative coding
    ("https://dev.to/t/creativecode/feed", "creative"),
    ("https://www.shadertoy.com/feed", "creative"),
    ("https://p5js.org/blog/feed.xml", "creative"),
    # Digital art
    ("https://www.behance.net/feeds/projects", "art"),
    ("https://www.deviantart.com/rss", "art"),
    ("https://www.artstation.com/blogs/feed.rss", "art"),
    # Interesting tools
    ("https://producthunt.com/feed", "tools"),
    ("https://alternativeto.net/feed/", "tools"),
    ("https://betapage.co/feed", "tools"),
    # Open source
    ("https://github.com/trending/feed", "opensource"),
    ("https://github.com/blog/open-source.rss", "opensource"),
    # Internet culture
    ("https://www.theverge.com/rss/index.xml", "culture"),
    ("https://arstechnica.com/feed/", "culture"),
    ("https://www.wired.com/feed/rss", "culture"),
    ("https://kotaku.com/rss", "culture"),
]

CATEGORY_LABELS: dict[str, str] = {
    "indie": "Indie Projects",
    "webdev": "Web Development",
    "creative": "Creative Coding",
    "art": "Digital Art",
    "tools": "Interesting Tools",
    "opensource": "Open Source",
    "culture": "Internet Culture",
}


class RSSCollectorError(Exception):
    """Raised when the RSS collector encounters an unrecoverable error."""


class RSSCollector:
    """Collects URLs from RSS/Atom feeds.

    Maintains a list of feeds to monitor, fetches and parses them,
    and extracts article links with metadata.
    """

    def __init__(self) -> None:
        self._feeds: list[dict[str, str]] = []
        self._last_request_time: float = 0.0
        self._session: requests.Session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a requests session with sensible defaults."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "DailyRandomWebsite/1.0 "
                    "(RSS Collector; +https://github.com/daily-random-website)"
                ),
                "Accept": (
                    "application/rss+xml, application/xml, "
                    "application/atom+xml, text/xml, */*"
                ),
            }
        )
        session.max_redirects = MAX_REDIRECTS  # type: ignore[attr-defined]
        return session

    def add_feed(self, url: str, category: str) -> None:
        """Add an RSS feed URL to monitor.

        Args:
            url: The feed URL (RSS or Atom).
            category: The source category (e.g. 'webdev', 'indie').

        Raises:
            ValueError: If url is empty or category is invalid.
        """
        if not url or not url.strip():
            raise ValueError("Feed URL cannot be empty")
        if not category or not category.strip():
            raise ValueError("Category cannot be empty")
        # Don't add duplicates
        for feed in self._feeds:
            if feed["url"] == url:
                logger.debug("Feed already registered: %s", url)
                return
        self._feeds.append({"url": url.strip(), "category": category.strip()})
        logger.info("Added feed: %s [%s]", url, category)

    def load_curated_feeds(self) -> None:
        """Load the built-in curated feed list."""
        for url, category in CURATED_FEEDS:
            self.add_feed(url, category)
        logger.info("Loaded %d curated feeds", len(CURATED_FEEDS))

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < REQUEST_DELAY and self._last_request_time > 0:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    def fetch_feed(self, url: str) -> feedparser.FeedParserDict:
        """Fetch and parse a single RSS/Atom feed.

        Args:
            url: The feed URL to fetch.

        Returns:
            The parsed feed as a feedparser dict.

        Raises:
            RSSCollectorError: If the feed cannot be fetched or parsed.
        """
        self._rate_limit()

        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.TooManyRedirects:
            raise RSSCollectorError(
                f"Too many redirects fetching feed: {url}"
            )
        except requests.exceptions.ConnectionError as exc:
            raise RSSCollectorError(
                f"Connection error fetching feed {url}: {exc}"
            )
        except requests.exceptions.Timeout:
            raise RSSCollectorError(f"Timeout fetching feed: {url}")
        except requests.exceptions.HTTPError as exc:
            raise RSSCollectorError(
                f"HTTP {response.status_code} fetching feed {url}: {exc}"
            )
        except requests.exceptions.RequestException as exc:
            raise RSSCollectorError(
                f"Request error fetching feed {url}: {exc}"
            )

        try:
            feed = feedparser.parse(response.content)
        except Exception as exc:
            raise RSSCollectorError(
                f"Error parsing feed {url}: {exc}"
            ) from exc

        # Validate that we actually got a feed
        if feed.bozo and not feed.entries:
            error_msg = getattr(feed, "bozo_exception", "unknown parse error")
            raise RSSCollectorError(
                f"Invalid feed content from {url}: {error_msg}"
            )

        return feed

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if a URL is well-formed and uses http(s)."""
        if not url or not isinstance(url, str):
            return False
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except (ValueError, AttributeError):
            return False

    @staticmethod
    def _extract_links(entry: Any, feed_url: str) -> list[str]:
        """Extract external URLs from a feed entry.

        Looks for entry links, via links, and content links.
        Filters out the feed URL itself and non-http(s) URLs.

        Args:
            entry: A feedparser entry object.
            feed_url: The URL of the feed (to exclude).

        Returns:
            A list of valid external URLs found in the entry.
        """
        links: list[str] = []

        # Primary link
        if hasattr(entry, "link") and entry.link:
            links.append(entry.link)

        # Additional links from the links list
        if hasattr(entry, "links") and entry.links:
            for link_info in entry.links:
                href = getattr(link_info, "href", None) or link_info.get("href")
                if href:
                    links.append(href)

        # Links embedded in content (e.g. <a href="..."> in summary/content)
        for content_field in ("content", "summary"):
            content = getattr(entry, content_field, None)
            if content:
                text = content if isinstance(content, str) else ""
                # Check if content is a list of dicts (common in feedparser)
                if isinstance(content, list) and content:
                    text = content[0].get("value", "")
                # Simple extraction of hrefs from HTML
                import re

                hrefs = re.findall(r'href=["\']([^"\']+)["\']', text)
                links.extend(hrefs)

        # Deduplicate while preserving order, and filter out feed URL
        seen: set[str] = set()
        feed_domain = urlparse(feed_url).netloc
        unique: list[str] = []
        for link in links:
            # Normalize the link
            link = link.strip()
            if not RSSCollector._is_valid_url(link):
                continue
            # Skip the feed's own domain (likely just the homepage)
            parsed = urlparse(link)
            if parsed.netloc == feed_domain:
                continue
            if link not in seen:
                seen.add(link)
                unique.append(link)

        return unique

    def collect_all(self, limit: int = MAX_URLS) -> list[dict[str, str]]:
        """Fetch all registered feeds and return collected URLs.

        Args:
            limit: Maximum number of URLs to return (default 100).

        Returns:
            A list of dicts, each with keys:
            - url: The article/resource URL
            - source: Always "rss"
            - source_type: The feed category
            - title: Article title
            - description: Article description/summary
        """
        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for feed_info in self._feeds:
            if len(results) >= limit:
                break

            feed_url = feed_info["url"]
            category = feed_info["category"]

            try:
                feed = self.fetch_feed(feed_url)
            except RSSCollectorError as exc:
                logger.warning("Skipping feed %s: %s", feed_url, exc)
                continue

            for entry in feed.entries:
                if len(results) >= limit:
                    break

                # Get article-level links
                article_links = self._extract_links(entry, feed_url)

                if not article_links:
                    # If no external links, fall back to entry.link
                    if hasattr(entry, "link") and entry.link:
                        article_links = [entry.link]
                    else:
                        continue

                # Use the first (best) link
                url = article_links[0]

                # Skip duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Extract metadata
                title = getattr(entry, "title", "") or ""
                description = getattr(entry, "summary", "") or getattr(
                    entry, "description", ""
                ) or ""
                # Truncate description to a reasonable length
                if len(description) > 500:
                    description = description[:497] + "..."

                results.append(
                    {
                        "url": url,
                        "source": "rss",
                        "source_type": category,
                        "title": str(title).strip(),
                        "description": description.strip(),
                    }
                )

        logger.info(
            "Collected %d URLs from %d feeds", len(results), len(self._feeds)
        )
        return results


def main() -> None:
    """CLI entry point: fetch all feeds and print collected URLs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    collector = RSSCollector()
    collector.load_curated_feeds()

    print(f"Registered {len(collector._feeds)} feeds")
    print("Fetching feeds (this may take a minute due to rate limiting)...\n")

    results = collector.collect_all(limit=MAX_URLS)

    print(f"\n{'='*60}")
    print(f"Collected {len(results)} URLs from RSS feeds")
    print(f"{'='*60}\n")

    for i, item in enumerate(results, 1):
        print(f"{i:3d}. [{item['source_type']}] {item['title'][:70]}")
        print(f"     {item['url']}")
        if item["description"]:
            desc = item["description"][:120]
            print(f"     {desc}...")
        print()

    if len(results) < 10:
        print(
            f"\nWARNING: Only collected {len(results)} URLs "
            f"(expected >= 10 for acceptance)"
        )


if __name__ == "__main__":
    main()
