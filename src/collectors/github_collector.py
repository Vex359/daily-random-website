"""GitHub trending repository collector.

Fetches trending and notable repositories from GitHub's API
to surface interesting creative coding, interactive, and web experiment projects.
Uses the GitHub Search API (REST) without authentication (60 req/hr rate limit).
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
SEARCH_ENDPOINT = f"{GITHUB_API_BASE}/search/repositories"

# Search query categories and their search terms
SEARCH_QUERIES: dict[str, list[str]] = {
    "creative_coding": [
        "creative coding",
        "generative art",
        "shadertoy",
    ],
    "interactive": [
        "interactive demo",
        "web experiment",
        "interactive visualization",
    ],
    "web_experiment": [
        "small web",
        "indie game",
        "experimental web",
    ],
}

# User-Agent header (GitHub API requires it)
DEFAULT_USER_AGENT = "daily-random-website-collector"


class GitHubCollectorError(Exception):
    """Base exception for GitHub collector errors."""


class RateLimitError(GitHubCollectorError):
    """Raised when GitHub API rate limit is exceeded."""


class GitHubCollector:
    """Collects interesting repository homepage URLs from GitHub's Search API.

    Operates without authentication (60 requests/hour limit).
    Focuses on repositories that have a ``homepage`` URL populated.
    """

    def __init__(self, token: str | None = None, timeout: int = 30) -> None:
        """Initialise the collector.

        Args:
            token: Optional GitHub personal access token. When provided the
                rate limit increases to 5 000 req/hr and the request is
                sent with an ``Authorization`` header.
            timeout: HTTP timeout in seconds for each request.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "User-Agent": DEFAULT_USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

        # Simple rate-limit tracking
        self._remaining: int = 60
        self._reset_at: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_repositories(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search GitHub repositories by query string.

        Args:
            query: The search query (e.g. ``"creative coding"``).
            limit: Maximum number of results to return (capped at 100 per
                GitHub API page; internally paginates if needed).

        Returns:
            List of repository dicts from the GitHub API.
        """
        self._check_rate_limit()

        per_page = min(limit, 100)
        params: dict[str, Any] = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": per_page,
        }

        repos: list[dict[str, Any]] = []
        page = 1

        while len(repos) < limit:
            params["page"] = page
            params["per_page"] = min(per_page, limit - len(repos))

            try:
                response = self.session.get(
                    SEARCH_ENDPOINT,
                    params=params,
                    timeout=self.timeout,
                )
                self._update_rate_limit(response)

                if response.status_code == 403:
                    raise RateLimitError(
                        "GitHub API rate limit exceeded. "
                        "Reset at: "
                        f"{time.ctime(self._reset_at)}"
                    )

                response.raise_for_status()

                data = response.json()
                items = data.get("items", [])
                if not items:
                    break

                repos.extend(items)
                page += 1

                # Stop if we have enough or hit the 1 000-result cap
                if len(repos) >= limit or page * per_page >= 1000:
                    break

            except requests.exceptions.Timeout:
                logger.warning("Timeout fetching GitHub repos for query: %s", query)
                break
            except requests.exceptions.ConnectionError:
                logger.warning("Connection error for query: %s", query)
                break
            except requests.exceptions.HTTPError as exc:
                logger.warning("HTTP error for query %s: %s", query, exc)
                break

        return repos[:limit]

    def collect_creative_coding(self, limit: int = 50) -> list[dict[str, Any]]:
        """Collect repositories related to creative coding.

        Searches for generative art, shader experiments, and similar projects.

        Args:
            limit: Maximum number of URL dicts to return.

        Returns:
            List of collected content dicts with homepage URLs.
        """
        return self._collect_by_category("creative_coding", limit)

    def collect_interactive(self, limit: int = 50) -> list[dict[str, Any]]:
        """Collect repositories with interactive demos or visualisations.

        Args:
            limit: Maximum number of URL dicts to return.

        Returns:
            List of collected content dicts with homepage URLs.
        """
        return self._collect_by_category("interactive", limit)

    def collect_all(self, limit: int = 150) -> list[dict[str, Any]]:
        """Collect from all search categories and return combined results.

        Distributes the ``limit`` across categories proportionally, then
        deduplicates by URL.

        Args:
            limit: Total maximum number of URL dicts to return.

        Returns:
            Deduplicated list of content dicts.
        """
        categories = list(SEARCH_QUERIES.keys())
        per_category = max(limit // len(categories), 1)

        all_results: list[dict[str, Any]] = []
        for category in categories:
            results = self._collect_by_category(category, per_category)
            all_results.extend(results)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in all_results:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(item)

        return deduped[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_by_category(
        self, category: str, limit: int
    ) -> list[dict[str, Any]]:
        """Run all queries for a category and extract homepage URLs."""
        queries = SEARCH_QUERIES.get(category, [])
        if not queries:
            return []

        per_query = max(limit // len(queries), 1)
        results: list[dict[str, Any]] = []

        for query in queries:
            repos = self.search_repositories(query, limit=per_query)
            for repo in repos:
                item = self._repo_to_item(repo, category)
                if item:
                    results.append(item)

        # Deduplicate within category by URL
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in results:
            url = item["url"]
            if url not in seen:
                seen.add(url)
                deduped.append(item)

        return deduped[:limit]

    @staticmethod
    def _repo_to_item(repo: dict[str, Any], source_type: str) -> dict[str, Any] | None:
        """Convert a GitHub repo dict to a pipeline item dict.

        Returns ``None`` if the repo has no usable homepage URL.
        """
        homepage = (repo.get("homepage") or "").strip()
        if not homepage:
            return None

        # Basic URL validation – must start with http(s)
        if not homepage.startswith(("http://", "https://")):
            homepage = "https://" + homepage

        return {
            "url": homepage,
            "source": "github",
            "source_type": source_type,
            "title": repo.get("name", "Untitled"),
            "description": repo.get("description") or "",
        }

    def _check_rate_limit(self) -> None:
        """Raise if we have exhausted the rate limit."""
        if self._remaining <= 0:
            now = time.time()
            if now < self._reset_at:
                wait = self._reset_at - now
                raise RateLimitError(
                    f"Rate limit exhausted. Try again in {wait:.0f}s."
                )
            # Reset window passed; allow requests again
            self._remaining = 60

    def _update_rate_limit(self, response: requests.Response) -> None:
        """Parse ``X-RateLimit-*`` headers from the response."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            self._remaining = int(remaining)
        if reset is not None:
            self._reset_at = float(reset)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    collector = GitHubCollector()
    results = collector.collect_all(limit=150)

    print(f"\nCollected {len(results)} URLs:\n")
    for i, item in enumerate(results, 1):
        title = item["title"].encode("ascii", "replace").decode()
        url = item["url"]
        desc = item["description"][:80].encode("ascii", "replace").decode()
        print(f"{i:>3}. [{item['source_type']}] {title}")
        print(f"     {url}")
        if desc:
            print(f"     {desc}")
        print()

    if len(results) < 10:
        logger.warning(
            "Only collected %d URLs (expected >= 10). "
            "GitHub rate-limit may be in effect.",
            len(results),
        )


if __name__ == "__main__":
    _main()
