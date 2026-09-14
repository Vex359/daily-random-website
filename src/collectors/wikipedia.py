"""Wikipedia random article collector.

Fetches random Wikipedia articles to surface interesting
encyclopedic content for the daily website.

Uses the MediaWiki API to get random articles, then parses
the rendered HTML to extract external links (non-Wikipedia).

Rate limiting: 1 request per second (respectful of Wikipedia).
"""

from __future__ import annotations

import re
import time
import urllib.parse
from typing import Any

import requests
from bs4 import BeautifulSoup

# Domains to exclude — mainstream/social media we never want to surface
EXCLUDED_DOMAINS: set[str] = {
    "wikipedia.org",
    "www.wikipedia.org",
    "en.wikipedia.org",
    "commons.wikimedia.org",
    "mediawiki.org",
    "www.mediawiki.org",
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
    "reddit.com",
    "www.reddit.com",
    "linkedin.com",
    "www.linkedin.com",
    "pinterest.com",
    "www.pinterest.com",
    "tumblr.com",
    "www.tumblr.com",
    "snapchat.com",
    "www.snapchat.com",
    "whatsapp.com",
    "www.whatsapp.com",
    "t.me",
    "telegram.org",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY = 1.0  # seconds between requests


def _extract_domain(url: str) -> str | None:
    """Extract the root domain from a URL (e.g. 'example.com' from 'https://www.example.com/foo')."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        # Strip leading 'www.'
        if host.startswith("www."):
            host = host[4:]
        return host.lower() if host else None
    except Exception:
        return None


def _is_excluded(url: str) -> bool:
    """Return True if the URL belongs to an excluded domain."""
    domain = _extract_domain(url)
    if domain is None:
        return True
    # Check exact match and with www prefix
    return domain in EXCLUDED_DOMAINS or f"www.{domain}" in EXCLUDED_DOMAINS


class WikipediaCollector:
    """Collects external URLs from random Wikipedia articles.

    Usage:
        collector = WikipediaCollector()
        urls = collector.collect_all(limit=50)
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "DailyRandomWebsite/1.0 (educational project; Python/requests)",
                "Accept": "application/json",
            }
        )
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """Make a rate-limited GET request."""
        self._rate_limit()
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fetch_random_articles(self, limit: int = 10) -> list[dict[str, str]]:
        """Fetch random Wikipedia articles via the MediaWiki API.

        Returns a list of dicts with keys:
            - pageid: int
            - title: str
            - url: str  (canonical article URL)

        The API caps ``rnlimit`` at 50 per request.
        """
        articles: list[dict[str, str]] = []
        remaining = limit

        while remaining > 0:
            batch = min(remaining, 50)
            params = {
                "action": "query",
                "list": "random",
                "rnnamespace": "0",  # main namespace only
                "rnlimit": str(batch),
                "format": "json",
            }
            resp = self._get(WIKI_API, params=params)
            data = resp.json()

            for item in data.get("query", {}).get("random", []):
                title = item["title"]
                encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="/:@!$&'()*+,;=")
                article_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
                articles.append(
                    {
                        "pageid": str(item["id"]),
                        "title": title,
                        "url": article_url,
                    }
                )

            remaining -= batch
            # If the API returned fewer than requested, we've exhausted the pool
            if len(data.get("query", {}).get("random", [])) < batch:
                break

        return articles

    def extract_external_links(self, article_html: str) -> list[str]:
        """Extract external links from rendered Wikipedia article HTML.

        Parses ``<a>`` tags in the article body, keeping only links to
        external sites (not internal Wikipedia links).

        Returns a deduplicated list of absolute URLs.
        """
        soup = BeautifulSoup(article_html, "html.parser")
        seen: set[str] = set()
        links: list[str] = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Skip internal / relative / fragment-only links
            if not href.startswith("http"):
                continue

            # Skip Wikipedia-internal links (multiple patterns)
            parsed = urllib.parse.urlparse(href)
            host = (parsed.hostname or "").lower()
            if any(
                host.endswith(domain)
                for domain in (
                    "wikipedia.org",
                    "wikimedia.org",
                    "mediawiki.org",
                )
            ):
                continue

            # Skip excluded mainstream domains
            if _is_excluded(href):
                continue

            # Normalise
            clean = href.rstrip("/")
            if clean not in seen:
                seen.add(clean)
                links.append(clean)

        return links

    def collect_all(self, limit: int = 50) -> list[dict[str, str]]:
        """Collect external URLs from random Wikipedia articles.

        Fetches random articles (up to ``limit``), extracts external
        links from each, and returns a flat list of result dicts.

        Each dict contains:
            - url: the external URL
            - source: "wikipedia"
            - title: Wikipedia article title the URL was found in
            - article_url: link to the Wikipedia article itself

        The total number of returned URLs is capped at ``limit``.
        """
        # Fetch enough articles to get a good spread of external links.
        # Each article typically has 10-30 external links, so requesting
        # fewer articles than the URL limit is usually sufficient.
        num_articles = max(1, limit // 3)
        articles = self.fetch_random_articles(limit=num_articles)

        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for article in articles:
            if len(results) >= limit:
                break

            # Fetch rendered HTML of the article
            parse_params = {
                "action": "parse",
                "page": article["title"],
                "prop": "text",
                "format": "json",
                "disabletoc": "1",
                "disableeditsection": "1",
            }
            try:
                resp = self._get(WIKI_API, params=parse_params)
                data = resp.json()
                html = data.get("parse", {}).get("text", {}).get("*", "")
            except (requests.RequestException, KeyError, ValueError):
                # Skip articles that fail to load
                continue

            ext_links = self.extract_external_links(html)

            for link in ext_links:
                if len(results) >= limit:
                    break
                if link not in seen_urls:
                    seen_urls.add(link)
                    results.append(
                        {
                            "url": link,
                            "source": "wikipedia",
                            "title": article["title"],
                            "article_url": article["url"],
                        }
                    )

        return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    collector = WikipediaCollector()
    urls = collector.collect_all(limit=50)
    print(f"\n{'='*60}")
    print(f"  Wikipedia Collector — found {len(urls)} external URLs")
    print(f"{'='*60}\n")
    for i, item in enumerate(urls, 1):
        print(f"  {i:3d}. {item['url']}")
        print(f"       from: {item['title']}")
    print()
