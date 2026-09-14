"""Awesome list collector.

Discovers and parses curated awesome lists on GitHub
to find high-quality resources and tools.

Uses the GitHub Search API (no authentication required) to find
awesome-* repositories, then fetches and parses their README.md
files to extract curated URLs.

Rate limiting: 1 request per second (GitHub unauthenticated limit
is ~10 req/min for search, ~60 for content).
"""

from __future__ import annotations

import re
import time
import urllib.parse
from typing import Any

import requests

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
REQUEST_DELAY = 1.0  # seconds between requests

# Domains to exclude — GitHub-internal or social media we never want
EXCLUDED_DOMAINS: set[str] = {
    "github.com",
    "www.github.com",
    "github.io",
    "raw.githubusercontent.com",
    "gist.github.com",
    "api.github.com",
    "www.youtube.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "reddit.com",
    "www.reddit.com",
    "linkedin.com",
    "www.linkedin.com",
}


def _extract_domain(url: str) -> str | None:
    """Extract the root domain from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
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
    return domain in EXCLUDED_DOMAINS or f"www.{domain}" in EXCLUDED_DOMAINS


# Regex to match Markdown links: [text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")


class AwesomeListsCollector:
    """Collects external URLs from GitHub awesome lists.

    Usage:
        collector = AwesomeListsCollector()
        urls = collector.collect_all(limit=50)
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "DailyRandomWebsite/1.0 (educational project; Python/requests)",
                "Accept": "application/vnd.github.v3+json",
            }
        )
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        """Make a rate-limited GET request."""
        self._rate_limit()
        resp = self.session.get(url, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def search_awesome_repos(self, query: str = "awesome", limit: int = 10) -> list[dict[str, str]]:
        """Search GitHub for awesome-* repositories.

        Uses the GitHub Search API to find repositories whose name
        or description contains the given query.

        Returns a list of dicts with keys:
            - full_name: e.g. "sindresorhus/awesome-nodejs"
            - description: repo description (may be empty)
            - url: HTML URL of the repo
            - default_branch: default branch name (usually "main" or "master")
        """
        repos: list[dict[str, str]] = []
        remaining = limit
        page = 1

        while remaining > 0:
            batch = min(remaining, 30)  # GitHub search max per page is 30
            params = {
                "q": f"{query} in:name",
                "sort": "stars",
                "order": "desc",
                "per_page": str(batch),
                "page": str(page),
            }
            try:
                resp = self._get(f"{GITHUB_API}/search/repositories", params=params)
                data = resp.json()
            except (requests.RequestException, ValueError):
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                repos.append(
                    {
                        "full_name": item["full_name"],
                        "description": item.get("description") or "",
                        "url": item["html_url"],
                        "default_branch": item.get("default_branch", "main"),
                    }
                )
                remaining -= 1
                if remaining <= 0:
                    break

            page += 1

        return repos

    def extract_urls_from_readme(self, repo: dict[str, str]) -> list[str]:
        """Fetch and parse the README.md of a GitHub repository.

        Extracts all external URLs from Markdown links and raw
        URLs in the text.

        Returns a deduplicated list of URLs, excluding GitHub-internal
        and other excluded domains.
        """
        full_name = repo["full_name"]
        branch = repo.get("default_branch", "main")
        raw_url = f"{GITHUB_RAW}/{full_name}/{branch}/README.md"

        try:
            resp = self._get(raw_url)
            content = resp.text
        except requests.RequestException:
            return []

        return self._parse_readme_content(content)

    def _parse_readme_content(self, content: str) -> list[str]:
        """Parse Markdown content and extract external URLs.

        Handles both Markdown link syntax and bare URLs.
        """
        seen: set[str] = set()
        links: list[str] = []

        # Extract from Markdown links: [text](url)
        for match in _MD_LINK_RE.finditer(content):
            url = match.group(2).strip()
            self._add_url(url, seen, links)

        # Also find bare URLs not wrapped in Markdown syntax
        bare_url_re = re.compile(r"(?<!\()(https?://[^\s\)\]\"'>]+)")
        for match in bare_url_re.finditer(content):
            url = match.group(1).strip().rstrip(".,;:!?)")
            self._add_url(url, seen, links)

        return links

    def _add_url(self, url: str, seen: set[str], links: list[str]) -> None:
        """Add a URL to the list if it's valid and not excluded."""
        # Normalise trailing slash
        clean = url.rstrip("/")
        if clean in seen:
            return
        if _is_excluded(clean):
            return
        # Must be a valid HTTP(S) URL
        parsed = urllib.parse.urlparse(clean)
        if parsed.scheme not in ("http", "https"):
            return
        seen.add(clean)
        links.append(clean)

    def collect_all(self, limit: int = 50) -> list[dict[str, str]]:
        """Collect external URLs from awesome lists on GitHub.

        Searches for awesome-* repos, fetches each README, and
        extracts external URLs. Returns a flat list of result dicts.

        Each dict contains:
            - url: the external URL
            - source: "awesome_lists"
            - repo: full_name of the awesome list repo
            - repo_url: HTML URL of the repo

        The total number of returned URLs is capped at ``limit``.
        """
        repos = self.search_awesome_repos(query="awesome", limit=15)

        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        for repo in repos:
            if len(results) >= limit:
                break

            ext_urls = self.extract_urls_from_readme(repo)

            for url in ext_urls:
                if len(results) >= limit:
                    break
                if url not in seen_urls:
                    seen_urls.add(url)
                    results.append(
                        {
                            "url": url,
                            "source": "awesome_lists",
                            "repo": repo["full_name"],
                            "repo_url": repo["url"],
                        }
                    )

        return results


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    collector = AwesomeListsCollector()
    urls = collector.collect_all(limit=50)
    print(f"\n{'='*60}")
    print(f"  Awesome Lists Collector — found {len(urls)} external URLs")
    print(f"{'='*60}\n")
    for i, item in enumerate(urls, 1):
        print(f"  {i:3d}. {item['url']}")
        print(f"       from: {item['repo']}")
    print()
