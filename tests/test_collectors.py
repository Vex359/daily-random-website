"""Tests for content collectors.

Verifies that each collector correctly fetches and parses
content from its respective source. All external HTTP calls
are mocked — no real API requests are made during tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.collectors.wikipedia import (
    WikipediaCollector,
    _extract_domain,
    _is_excluded,
)
from src.collectors.awesome_lists import (
    AwesomeListsCollector,
    _extract_domain as aw_extract_domain,
    _is_excluded as aw_is_excluded,
)


# ======================================================================
# Wikipedia helper tests
# ======================================================================


class TestWikipediaHelpers:
    """Tests for the module-level helper functions in wikipedia.py."""

    def test_extract_domain_basic(self):
        assert _extract_domain("https://www.example.com/page") == "example.com"

    def test_extract_domain_no_www(self):
        assert _extract_domain("https://example.com/page") == "example.com"

    def test_extract_domain_subdomain(self):
        assert _extract_domain("https://blog.example.com") == "blog.example.com"

    def test_extract_domain_invalid(self):
        assert _extract_domain("not-a-url") is None

    def test_extract_domain_empty(self):
        assert _extract_domain("") is None

    def test_is_excluded_wikipedia(self):
        assert _is_excluded("https://en.wikipedia.org/wiki/Python") is True

    def test_is_excluded_youtube(self):
        assert _is_excluded("https://www.youtube.com/watch?v=abc") is True

    def test_is_excluded_twitter(self):
        assert _is_excluded("https://twitter.com/user/status/123") is True

    def test_is_excluded_facebook(self):
        assert _is_excluded("https://facebook.com/page") is True

    def test_is_excluded_valid(self):
        assert _is_excluded("https://www.python.org/doc") is False

    def test_is_excluded_trailing_slash(self):
        assert _is_excluded("https://en.wikipedia.org/") is True

    def test_is_excluded_invalid_url(self):
        assert _is_excluded("not-a-url") is True


# ======================================================================
# WikipediaCollector tests
# ======================================================================


def _mock_wiki_random_response(titles: list[str]) -> dict:
    """Build a fake MediaWiki ``action=query&list=random`` response."""
    return {
        "query": {
            "random": [
                {"id": i + 100, "title": title}
                for i, title in enumerate(titles)
            ]
        }
    }


def _mock_wiki_parse_response(html: str) -> dict:
    """Build a fake MediaWiki ``action=parse`` response."""
    return {"parse": {"text": {"*": html}}}


class TestWikipediaCollector:
    """Tests for WikipediaCollector."""

    def _make_collector(self) -> WikipediaCollector:
        collector = WikipediaCollector()
        collector._last_request_time = 0.0  # skip rate limit wait
        collector.session = MagicMock()
        return collector

    # -- fetch_random_articles --

    def test_fetch_random_articles_returns_expected_count(self):
        collector = self._make_collector()
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_random_response(
            ["Python (programming language)", "Solar eclipse", "Cat"]
        )
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        articles = collector.fetch_random_articles(limit=3)
        assert len(articles) == 3
        assert articles[0]["title"] == "Python (programming language)"
        assert articles[0]["url"].startswith("https://en.wikipedia.org/wiki/")
        assert articles[0]["pageid"] == "100"

    def test_fetch_random_articles_batches_requests(self):
        collector = self._make_collector()
        titles = [f"Article {i}" for i in range(12)]
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_random_response(titles)
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        articles = collector.fetch_random_articles(limit=12)
        assert len(articles) == 12
        # Should have made 1 request (12 < 50 batch size)
        assert collector.session.get.call_count == 1

    def test_fetch_random_articles_handles_api_error(self):
        collector = self._make_collector()
        collector.session.get.side_effect = requests.RequestException("timeout")

        with pytest.raises(requests.RequestException):
            collector.fetch_random_articles(limit=5)

    # -- extract_external_links --

    def test_extract_external_links_basic(self):
        collector = self._make_collector()
        html = """
        <div>
            <a href="https://www.python.org">Python</a>
            <a href="https://docs.rs/futures">Futures crate</a>
            <a href="/wiki/Internal_link">Internal</a>
            <a href="https://en.wikipedia.org/wiki/Other_article">Wiki link</a>
        </div>
        """
        links = collector.extract_external_links(html)
        assert "https://www.python.org" in links
        assert "https://docs.rs/futures" in links
        # Internal / excluded links should not appear
        assert all("wikipedia.org" not in l for l in links)
        assert all("Internal" not in l for l in links)

    def test_extract_external_links_filters_excluded_domains(self):
        collector = self._make_collector()
        html = """
        <div>
            <a href="https://www.youtube.com/watch?v=abc">YouTube</a>
            <a href="https://twitter.com/user">Twitter</a>
            <a href="https://facebook.com/page">Facebook</a>
            <a href="https://www.python.org">Python</a>
        </div>
        """
        links = collector.extract_external_links(html)
        assert "https://www.python.org" in links
        assert len(links) == 1

    def test_extract_external_links_deduplication(self):
        collector = self._make_collector()
        html = """
        <a href="https://example.com">Link 1</a>
        <a href="https://example.com/">Link 2</a>
        """
        links = collector.extract_external_links(html)
        # Should be deduplicated (trailing slash stripped)
        assert links.count("https://example.com") == 1 or links.count("https://example.com/") == 1

    def test_extract_external_links_empty(self):
        collector = self._make_collector()
        assert collector.extract_external_links("") == []
        assert collector.extract_external_links("<div>no links</div>") == []

    # -- collect_all --

    def test_collect_all_returns_correct_shape(self):
        collector = self._make_collector()

        # Mock fetch_random_articles
        articles = [
            {"pageid": "1", "title": "Python", "url": "https://en.wikipedia.org/wiki/Python"},
        ]
        collector.fetch_random_articles = MagicMock(return_value=articles)

        # Mock the parse API response
        html = '<a href="https://www.python.org">Python</a><a href="https://docs.python.org">Docs</a>'
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_parse_response(html)
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        results = collector.collect_all(limit=10)
        assert len(results) == 2
        for item in results:
            assert "url" in item
            assert item["source"] == "wikipedia"
            assert "title" in item
            assert "article_url" in item

    def test_collect_all_respects_limit(self):
        collector = self._make_collector()
        articles = [
            {"pageid": str(i), "title": f"Article {i}", "url": f"https://en.wikipedia.org/wiki/Article_{i}"}
            for i in range(5)
        ]
        collector.fetch_random_articles = MagicMock(return_value=articles)

        html = (
            '<a href="https://example.com/1">1</a>'
            '<a href="https://example.com/2">2</a>'
            '<a href="https://example.com/3">3</a>'
        )
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_parse_response(html)
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        results = collector.collect_all(limit=2)
        assert len(results) == 2

    def test_collect_all_handles_parse_failure(self):
        collector = self._make_collector()
        articles = [
            {"pageid": "1", "title": "Broken", "url": "https://en.wikipedia.org/wiki/Broken"},
        ]
        collector.fetch_random_articles = MagicMock(return_value=articles)
        collector.session.get.side_effect = requests.RequestException("fail")

        results = collector.collect_all(limit=10)
        assert results == []


# ======================================================================
# Awesome Lists helper tests
# ======================================================================


class TestAwesomeListsHelpers:
    """Tests for the module-level helper functions in awesome_lists.py."""

    def test_extract_domain(self):
        assert aw_extract_domain("https://raw.githubusercontent.com/foo/bar/main/README.md") == "raw.githubusercontent.com"

    def test_extract_domain_empty(self):
        assert aw_extract_domain("") is None

    def test_is_excluded_github(self):
        assert aw_is_excluded("https://github.com/foo/bar") is True

    def test_is_excluded_githubusercontent(self):
        assert aw_is_excluded("https://raw.githubusercontent.com/foo/bar/main/x") is True

    def test_is_excluded_valid(self):
        assert aw_is_excluded("https://www.rust-lang.org") is False


# ======================================================================
# AwesomeListsCollector tests
# ======================================================================


def _mock_github_search_response(repo_names: list[str]) -> dict:
    """Build a fake GitHub Search API response."""
    return {
        "items": [
            {
                "full_name": name,
                "description": f"Awesome list: {name}",
                "html_url": f"https://github.com/{name}",
                "default_branch": "main",
            }
            for name in repo_names
        ]
    }


class TestAwesomeListsCollector:
    """Tests for AwesomeListsCollector."""

    def _make_collector(self) -> AwesomeListsCollector:
        collector = AwesomeListsCollector()
        collector._last_request_time = 0.0
        collector.session = MagicMock()
        return collector

    # -- search_awesome_repos --

    def test_search_awesome_repos_returns_expected_count(self):
        collector = self._make_collector()
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_github_search_response(
            ["sindresorhus/awesome-nodejs", "emijrp/awesome-ai", "jondot/awesome-react"]
        )
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        repos = collector.search_awesome_repos(limit=3)
        assert len(repos) == 3
        assert repos[0]["full_name"] == "sindresorhus/awesome-nodejs"
        assert repos[0]["url"] == "https://github.com/sindresorhus/awesome-nodejs"

    def test_search_awesome_repos_handles_empty(self):
        collector = self._make_collector()
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        repos = collector.search_awesome_repos(limit=10)
        assert repos == []

    def test_search_awesome_repos_handles_api_error(self):
        collector = self._make_collector()
        collector.session.get.side_effect = requests.RequestException("rate limit")

        repos = collector.search_awesome_repos(limit=10)
        assert repos == []

    # -- extract_urls_from_readme --

    def test_extract_urls_from_readme_basic(self):
        collector = self._make_collector()
        readme_content = """
        # Awesome Python

        - [Django](https://www.djangoproject.com/) - web framework
        - [Flask](https://flask.palletsprojects.com/) - micro framework
        - [GitHub Repo](https://github.com/someone/repo) - excluded
        """
        collector._parse_readme_content = MagicMock(
            return_value=["https://www.djangoproject.com/", "https://flask.palletsprojects.com/"]
        )

        repo = {
            "full_name": "test/awesome-python",
            "description": "test",
            "url": "https://github.com/test/awesome-python",
            "default_branch": "main",
        }

        # Mock the HTTP call
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.text = readme_content
        mock_resp.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_resp)

        urls = collector.extract_urls_from_readme(repo)
        assert len(urls) == 2
        assert "https://www.djangoproject.com/" in urls
        assert "https://flask.palletsprojects.com/" in urls

    def test_extract_urls_from_readme_filters_github(self):
        collector = self._make_collector()
        repo = {
            "full_name": "test/repo",
            "description": "",
            "url": "https://github.com/test/repo",
            "default_branch": "main",
        }
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.text = """
        [Link1](https://www.python.org)
        [Link2](https://github.com/user/project)
        [Link3](https://raw.githubusercontent.com/user/project/main/file)
        """
        mock_resp.raise_for_status = MagicMock()
        collector.session.get.return_value = mock_resp

        urls = collector.extract_urls_from_readme(repo)
        assert "https://www.python.org" in urls
        assert not any("github.com" in u for u in urls)

    def test_extract_urls_from_readme_handles_fetch_error(self):
        collector = self._make_collector()
        collector.session.get.side_effect = requests.RequestException("404")
        repo = {
            "full_name": "nonexistent/repo",
            "description": "",
            "url": "https://github.com/nonexistent/repo",
            "default_branch": "main",
        }
        urls = collector.extract_urls_from_readme(repo)
        assert urls == []

    # -- _parse_readme_content --

    def test_parse_readme_content_markdown_links(self):
        collector = self._make_collector()
        content = """
        # Title

        Visit [Example](https://example.com) for more info.
        Also see [Another](https://another.org/page?q=1).
        """
        urls = collector._parse_readme_content(content)
        assert "https://example.com" in urls
        assert "https://another.org/page?q=1" in urls

    def test_parse_readme_content_bare_urls(self):
        collector = self._make_collector()
        content = "Check out https://bare-url.com/path for info."
        urls = collector._parse_readme_content(content)
        assert "https://bare-url.com/path" in urls

    def test_parse_readme_content_excludes_domains(self):
        collector = self._make_collector()
        content = """
        [GitHub](https://github.com/user/repo)
        [YouTube](https://www.youtube.com/watch?v=abc)
        [Valid](https://example.com)
        """
        urls = collector._parse_readme_content(content)
        assert len(urls) == 1
        assert "https://example.com" in urls

    def test_parse_readme_content_deduplication(self):
        collector = self._make_collector()
        content = """
        [Link1](https://example.com)
        [Link2](https://example.com/)
        """
        urls = collector._parse_readme_content(content)
        # Deduplicated
        assert len(urls) == 1

    # -- collect_all --

    def test_collect_all_returns_correct_shape(self):
        collector = self._make_collector()
        repos = [
            {
                "full_name": "test/awesome-list",
                "description": "test",
                "url": "https://github.com/test/awesome-list",
                "default_branch": "main",
            }
        ]
        collector.search_awesome_repos = MagicMock(return_value=repos)
        collector.extract_urls_from_readme = MagicMock(
            return_value=["https://example.com/a", "https://example.com/b"]
        )

        results = collector.collect_all(limit=10)
        assert len(results) == 2
        for item in results:
            assert "url" in item
            assert item["source"] == "awesome_lists"
            assert item["repo"] == "test/awesome-list"
            assert item["repo_url"] == "https://github.com/test/awesome-list"

    def test_collect_all_respects_limit(self):
        collector = self._make_collector()
        repos = [
            {
                "full_name": f"test/repo{i}",
                "description": "",
                "url": f"https://github.com/test/repo{i}",
                "default_branch": "main",
            }
            for i in range(5)
        ]
        collector.search_awesome_repos = MagicMock(return_value=repos)
        side_effects = {f"test/repo{i}": [f"https://example.com/repo{i}"] for i in range(5)}

        def _extract(repo: dict) -> list[str]:
            return side_effects.get(repo["full_name"], [])

        collector.extract_urls_from_readme = MagicMock(side_effect=_extract)

        results = collector.collect_all(limit=3)
        assert len(results) == 3

    def test_collect_all_deduplicates_urls_across_repos(self):
        collector = self._make_collector()
        repos = [
            {
                "full_name": f"test/repo{i}",
                "description": "",
                "url": f"https://github.com/test/repo{i}",
                "default_branch": "main",
            }
            for i in range(3)
        ]
        collector.search_awesome_repos = MagicMock(return_value=repos)
        # Each repo returns the same URL
        collector.extract_urls_from_readme = MagicMock(
            return_value=["https://example.com/same"]
        )

        results = collector.collect_all(limit=50)
        assert len(results) == 1


# ======================================================================
# Integration-style tests (no real HTTP)
# ======================================================================


class TestWikipediaCollectorIntegration:
    """Higher-level tests verifying collector logic end-to-end with mocks."""

    def test_collect_all_filters_mainstream_domains(self):
        """Verify that mainstream domains are filtered out in collect_all."""
        collector = WikipediaCollector()
        collector._last_request_time = 0.0

        articles = [
            {"pageid": "1", "title": "Test", "url": "https://en.wikipedia.org/wiki/Test"},
        ]
        collector.fetch_random_articles = MagicMock(return_value=articles)

        html = """
        <a href="https://www.python.org">Python</a>
        <a href="https://www.youtube.com/watch?v=abc">YouTube</a>
        <a href="https://twitter.com/user">Twitter</a>
        <a href="https://en.wikipedia.org/wiki/Other">Internal</a>
        <a href="https://www.facebook.com/page">Facebook</a>
        """
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_parse_response(html)
        mock_resp.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_resp)

        results = collector.collect_all(limit=50)
        urls = [r["url"] for r in results]
        assert "https://www.python.org" in urls
        assert not any("youtube.com" in u for u in urls)
        assert not any("twitter.com" in u for u in urls)
        assert not any("facebook.com" in u for u in urls)
        assert not any("wikipedia.org" in u for u in urls)

    def test_collect_all_max_50_urls(self):
        """Verify collect_all never returns more than limit URLs."""
        collector = WikipediaCollector()
        collector._last_request_time = 0.0

        articles = [
            {"pageid": str(i), "title": f"Art{i}", "url": f"https://en.wikipedia.org/wiki/Art{i}"}
            for i in range(20)
        ]
        collector.fetch_random_articles = MagicMock(return_value=articles)

        html = '<a href="https://example.com/link">Link</a>'
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.json.return_value = _mock_wiki_parse_response(html)
        mock_resp.raise_for_status = MagicMock()
        collector.session.get = MagicMock(return_value=mock_resp)

        results = collector.collect_all(limit=50)
        assert len(results) <= 50


class TestAwesomeListsCollectorIntegration:
    """Higher-level tests verifying collector logic end-to-end with mocks."""

    def test_collect_all_filters_github_urls(self):
        """Verify that GitHub-internal URLs are filtered out by extract_urls_from_readme."""
        collector = AwesomeListsCollector()
        collector._last_request_time = 0.0

        repos = [
            {
                "full_name": "test/awesome-list",
                "description": "test",
                "url": "https://github.com/test/awesome-list",
                "default_branch": "main",
            }
        ]
        collector.search_awesome_repos = MagicMock(return_value=repos)
        # extract_urls_from_readme already filters github.com, so the mock
        # returns only valid external URLs (simulating what the real method does)
        collector.extract_urls_from_readme = MagicMock(
            return_value=[
                "https://www.python.org",
                "https://example.com/valid",
            ]
        )

        results = collector.collect_all(limit=50)
        urls = [r["url"] for r in results]
        assert "https://www.python.org" in urls
        assert "https://example.com/valid" in urls
        assert len(urls) == 2

    def test_collect_all_max_50_urls(self):
        """Verify collect_all never returns more than limit URLs."""
        collector = AwesomeListsCollector()
        collector._last_request_time = 0.0

        repos = [
            {
                "full_name": f"test/repo{i}",
                "description": "",
                "url": f"https://github.com/test/repo{i}",
                "default_branch": "main",
            }
            for i in range(20)
        ]
        collector.search_awesome_repos = MagicMock(return_value=repos)
        collector._parse_readme_content = MagicMock(
            return_value=["https://example.com/unique"]
        )

        results = collector.collect_all(limit=50)
        assert len(results) <= 50


# ======================================================================
# GitHub Collector tests
# ======================================================================

from src.collectors.github_collector import (
    GitHubCollector,
    GitHubCollectorError,
    RateLimitError,
)


def _gh_make_repo(
    name: str = "test-repo",
    homepage: str = "https://example.com",
    description: str = "A test repo",
) -> dict:
    """Create a minimal GitHub repo dict."""
    return {
        "name": name,
        "homepage": homepage,
        "description": description,
        "html_url": f"https://github.com/user/{name}",
        "stargazers_count": 100,
        "updated_at": "2025-01-01T00:00:00Z",
    }


def _gh_make_search_response(items: list[dict], total_count: int | None = None) -> dict:
    """Create a GitHub search API response body."""
    if total_count is None:
        total_count = len(items)
    return {
        "total_count": total_count,
        "incomplete_results": False,
        "items": items,
    }


def _gh_mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {
        "X-RateLimit-Remaining": "59",
        "X-RateLimit-Reset": "9999999999",
    }
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"HTTP {status_code}"
        )
    return resp


def _gh_empty_response() -> MagicMock:
    """Return a mock 200 response with no items (stops pagination)."""
    return _gh_mock_response(200, _gh_make_search_response([]))


class TestGitHubCollectorInit:
    """Test constructor and session setup."""

    def test_default_session_headers(self):
        collector = GitHubCollector()
        headers = collector.session.headers
        assert "Accept" in headers
        assert "User-Agent" in headers
        assert "X-GitHub-Api-Version" in headers
        assert "Authorization" not in headers

    def test_token_sets_auth_header(self):
        collector = GitHubCollector(token="ghp_test123")
        assert collector.session.headers["Authorization"] == "Bearer ghp_test123"


class TestGitHubSearchRepositories:
    """Test search_repositories method."""

    def test_basic_search(self):
        """Single page of results is returned correctly."""
        repo = _gh_make_repo()
        resp = _gh_mock_response(200, _gh_make_search_response([repo]))

        collector = GitHubCollector()
        with patch.object(
            collector.session, "get", side_effect=[resp, _gh_empty_response()]
        ) as mock_get:
            results = collector.search_repositories("creative coding", limit=50)

        assert len(results) == 1
        assert results[0]["name"] == "test-repo"

    def test_pagination(self):
        """Results are paginated when limit > per_page."""
        repos_page1 = [_gh_make_repo(name=f"repo-{i}") for i in range(50)]
        repos_page2 = [_gh_make_repo(name=f"repo-{i}") for i in range(50, 55)]

        resp1 = _gh_mock_response(200, _gh_make_search_response(repos_page1))
        resp2 = _gh_mock_response(200, _gh_make_search_response(repos_page2))

        collector = GitHubCollector()
        with patch.object(
            collector.session, "get", side_effect=[resp1, resp2]
        ) as mock_get:
            results = collector.search_repositories("test", limit=55)

        assert len(results) == 55
        assert mock_get.call_count == 2

    def test_empty_results(self):
        """Empty items list stops iteration."""
        resp = _gh_mock_response(200, _gh_make_search_response([]))
        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.search_repositories("nonexistent query")
        assert results == []

    def test_timeout_handled(self):
        """Timeout exceptions are caught and return partial results."""
        collector = GitHubCollector()
        with patch.object(
            collector.session, "get", side_effect=requests.exceptions.Timeout
        ):
            results = collector.search_repositories("test")
        assert results == []

    def test_connection_error_handled(self):
        """Connection errors are caught gracefully."""
        collector = GitHubCollector()
        with patch.object(
            collector.session, "get", side_effect=requests.exceptions.ConnectionError
        ):
            results = collector.search_repositories("test")
        assert results == []


class TestGitHubRateLimiting:
    """Test rate-limit tracking and enforcement."""

    def test_rate_limit_headers_parsed(self):
        """X-RateLimit headers update internal state."""
        resp = _gh_mock_response(
            200,
            _gh_make_search_response([_gh_make_repo()]),
            headers={"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "12345"},
        )
        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            collector.search_repositories("test")

        assert collector._remaining == 10
        assert collector._reset_at == 12345.0

    def test_rate_limit_exceeded_raises(self):
        """RateLimitError raised when remaining hits 0."""
        collector = GitHubCollector()
        collector._remaining = 0
        collector._reset_at = 9999999999.0  # far future

        with pytest.raises(RateLimitError, match="Rate limit exhausted"):
            collector.search_repositories("test")

    def test_rate_limit_reset_allows_requests(self):
        """After reset window passes, requests are allowed again."""
        collector = GitHubCollector()
        collector._remaining = 0
        collector._reset_at = 1.0  # past timestamp

        resp = _gh_mock_response(200, _gh_make_search_response([_gh_make_repo()]))
        with patch.object(
            collector.session, "get", side_effect=[resp, _gh_empty_response()]
        ):
            results = collector.search_repositories("test")

        assert len(results) == 1


class TestGitHubCollectCreativeCoding:
    """Test creative coding collection."""

    def test_collects_from_multiple_queries(self):
        """Results from all creative_coding queries are combined."""
        repos1 = [_gh_make_repo(name="gen-art-1", homepage="https://gen-art.example.com")]
        repos2 = [_gh_make_repo(name="shader-1", homepage="https://shader.example.com")]
        repos3 = [_gh_make_repo(name="creative-1", homepage="https://creative.example.com")]

        resp1 = _gh_mock_response(200, _gh_make_search_response(repos1))
        resp2 = _gh_mock_response(200, _gh_make_search_response(repos2))
        resp3 = _gh_mock_response(200, _gh_make_search_response(repos3))

        # 3 queries × 2 calls each (data + empty) = 6 responses
        collector = GitHubCollector()
        with patch.object(
            collector.session, "get", side_effect=[
                resp1, _gh_empty_response(),
                resp2, _gh_empty_response(),
                resp3, _gh_empty_response(),
            ]
        ):
            results = collector.collect_creative_coding(limit=10)

        assert len(results) == 3
        assert all(r["source_type"] == "creative_coding" for r in results)


class TestGitHubCollectInteractive:
    """Test interactive demo collection."""

    def test_collects_interactive_demos(self):
        repos = [_gh_make_repo(name="interactive-viz")]
        resp = _gh_mock_response(200, _gh_make_search_response(repos))

        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.collect_interactive(limit=10)

        assert len(results) >= 1
        assert all(r["source_type"] == "interactive" for r in results)


class TestGitHubCollectAll:
    """Test collect_all combines and deduplicates."""

    def test_deduplicates_by_url(self):
        """Same URL from different categories appears only once."""
        repo = _gh_make_repo(homepage="https://dup.example.com")

        resp = _gh_mock_response(200, _gh_make_search_response([repo]))
        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.collect_all(limit=50)

        urls = [r["url"] for r in results]
        assert urls.count("https://dup.example.com") == 1

    def test_respects_limit(self):
        """Total results never exceed the limit."""
        repos = [_gh_make_repo(name=f"r{i}", homepage=f"https://{i}.example.com") for i in range(20)]
        resp = _gh_mock_response(200, _gh_make_search_response(repos))

        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.collect_all(limit=10)

        assert len(results) <= 10

    def test_returns_correct_shape(self):
        """Each item has all required fields."""
        repo = _gh_make_repo(homepage="https://mysite.com", description="Cool project")
        resp = _gh_mock_response(200, _gh_make_search_response([repo]))

        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.collect_all(limit=10)

        assert len(results) >= 1
        item = results[0]
        assert item["url"] == "https://mysite.com"
        assert item["source"] == "github"
        assert "source_type" in item
        assert item["title"] == "test-repo"
        assert item["description"] == "Cool project"


class TestGitHubRepoToItem:
    """Test _repo_to_item conversion."""

    def test_valid_homepage(self):
        repo = _gh_make_repo(homepage="https://mysite.com")
        item = GitHubCollector._repo_to_item(repo, "interactive")
        assert item is not None
        assert item["url"] == "https://mysite.com"
        assert item["source"] == "github"
        assert item["source_type"] == "interactive"
        assert item["title"] == "test-repo"

    def test_no_homepage_returns_none(self):
        repo = _gh_make_repo(homepage="")
        item = GitHubCollector._repo_to_item(repo, "creative_coding")
        assert item is None

    def test_none_homepage_returns_none(self):
        repo = {"name": "x", "homepage": None, "description": "d"}
        item = GitHubCollector._repo_to_item(repo, "creative_coding")
        assert item is None

    def test_bare_domain_gets_prefix(self):
        repo = _gh_make_repo(homepage="mysite.com")
        item = GitHubCollector._repo_to_item(repo, "interactive")
        assert item is not None
        assert item["url"] == "https://mysite.com"

    def test_empty_description(self):
        repo = _gh_make_repo(homepage="https://ok.com", description=None)
        repo["description"] = None
        item = GitHubCollector._repo_to_item(repo, "interactive")
        assert item is not None
        assert item["description"] == ""


class TestGitHubHTTPErrorHandling:
    """Test various HTTP error scenarios."""

    def test_403_rate_limit_response(self):
        """403 response raises RateLimitError."""
        resp = _gh_mock_response(403)
        resp.raise_for_status.side_effect = None  # clear the generic error

        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            with pytest.raises(RateLimitError, match="rate limit exceeded"):
                collector.search_repositories("test")

    def test_422_validation_error(self):
        """422 returns empty results gracefully."""
        resp = _gh_mock_response(422)
        collector = GitHubCollector()
        with patch.object(collector.session, "get", return_value=resp):
            results = collector.search_repositories("invalid???query")
        assert isinstance(results, list)


# ======================================================================
# Hacker News Collector tests
# ======================================================================

from src.collectors.hackernews import (
    HackerNewsCollector,
    HackerNewsCollectorError,
    STORY_ENDPOINTS,
)


def _hn_make_item(
    item_id: int,
    url: str = "https://example.com",
    title: str = "Test Title",
    score: int = 100,
) -> dict:
    """Build a fake HN item dict."""
    return {"id": item_id, "title": title, "url": url, "score": score, "type": "story"}


def _hn_make_item_no_url(item_id: int, title: str = "Ask HN: foo") -> dict:
    """Build a fake HN item with no URL (e.g. Ask HN)."""
    return {"id": item_id, "title": title, "score": 10, "type": "story"}


def _hn_mock_response(json_data, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response for HN API."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestHackerNewsCollectorInit:
    """Test constructor defaults."""

    def test_default_timeout(self):
        collector = HackerNewsCollector()
        assert collector.timeout == 30

    def test_default_rate_limit(self):
        collector = HackerNewsCollector()
        assert collector.rate_limit_delay == 1.0

    def test_custom_params(self):
        collector = HackerNewsCollector(timeout=10, rate_limit_delay=0.5)
        assert collector.timeout == 10
        assert collector.rate_limit_delay == 0.5


class TestHackerNewsFetchStories:
    """Test the three main story-fetching methods."""

    def _make_collector(self) -> HackerNewsCollector:
        collector = HackerNewsCollector(rate_limit_delay=0)
        return collector

    # -- fetch_new_stories --

    def test_fetch_new_stories_returns_urls(self):
        collector = self._make_collector()
        items = [
            _hn_make_item(1, "https://a.com", "Story A"),
            _hn_make_item(2, "https://b.com", "Story B"),
        ]
        responses = {
            "/newstories.json": [1, 2],
            "/item/1.json": items[0],
            "/item/2.json": items[1],
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_new_stories(limit=2)

        assert len(results) == 2
        assert all(r["source_type"] == "new" for r in results)
        assert all(r["source"] == "hackernews" for r in results)
        assert results[0]["url"] == "https://a.com"
        collector.close()

    def test_fetch_new_stories_skips_items_without_url(self):
        collector = self._make_collector()
        responses = {
            "/newstories.json": [1, 2],
            "/item/1.json": _hn_make_item(1, "https://has-url.com"),
            "/item/2.json": _hn_make_item_no_url(2, "Ask HN: no url"),
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_new_stories(limit=10)
        assert len(results) == 1
        assert results[0]["url"] == "https://has-url.com"
        collector.close()

    def test_fetch_new_stories_handles_api_failure(self):
        collector = self._make_collector()
        with patch.object(collector, "_get_json", return_value=None):
            results = collector.fetch_new_stories()
        assert results == []
        collector.close()

    def test_fetch_new_stories_handles_non_list_response(self):
        collector = self._make_collector()
        with patch.object(collector, "_get_json", return_value="not-a-list"):
            results = collector.fetch_new_stories()
        assert results == []
        collector.close()

    # -- fetch_top_stories --

    def test_fetch_top_stories_returns_urls(self):
        collector = self._make_collector()
        responses = {
            "/topstories.json": [10],
            "/item/10.json": _hn_make_item(10, "https://top1.com", "Top 1", 500),
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_top_stories(limit=1)
        assert len(results) == 1
        assert results[0]["source_type"] == "top"
        assert results[0]["score"] == 500
        collector.close()

    def test_fetch_top_stories_handles_empty_response(self):
        collector = self._make_collector()
        with patch.object(collector, "_get_json", return_value=[]):
            results = collector.fetch_top_stories()
        assert results == []
        collector.close()

    # -- fetch_best_stories --

    def test_fetch_best_stories_returns_urls(self):
        collector = self._make_collector()
        responses = {
            "/beststories.json": [100, 200],
            "/item/100.json": _hn_make_item(100, "https://best1.com", "Best 1", 999),
            "/item/200.json": _hn_make_item(200, "https://best2.com", "Best 2", 888),
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_best_stories(limit=2)
        assert len(results) == 2
        assert all(r["source_type"] == "best" for r in results)
        collector.close()


class TestHackerNewsShowHN:
    """Test Show HN filtering."""

    def _make_collector(self) -> HackerNewsCollector:
        return HackerNewsCollector(rate_limit_delay=0)

    def test_filters_show_hn_titles(self):
        collector = self._make_collector()
        responses = {
            "/topstories.json": [1, 2, 3],
            "/item/1.json": _hn_make_item(1, "https://showhn.com", "Show HN: My Project", 200),
            "/item/2.json": _hn_make_item(2, "https://news.com", "Regular story", 300),
            "/item/3.json": _hn_make_item_no_url(3, "Show HN: No URL Project"),
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_show_hn(limit=10)
        assert len(results) == 1
        assert results[0]["source_type"] == "show_hn"
        assert "Show HN" in results[0]["title"]
        collector.close()

    def test_case_insensitive_matching(self):
        collector = self._make_collector()
        responses = {
            "/topstories.json": [1],
            "/item/1.json": _hn_make_item(1, "https://example.com", "show hn: lowercase", 50),
        }
        with patch.object(collector, "_get_json", side_effect=lambda p: responses.get(p)):
            results = collector.fetch_show_hn(limit=10)
        assert len(results) == 1
        assert results[0]["source_type"] == "show_hn"
        collector.close()

    def test_handles_no_stories(self):
        collector = self._make_collector()
        with patch.object(collector, "_get_json", return_value=None):
            results = collector.fetch_show_hn()
        assert results == []
        collector.close()


class TestHackerNewsCollectAll:
    """Test collect_all deduplication and limiting."""

    def _make_collector(self) -> HackerNewsCollector:
        return HackerNewsCollector(rate_limit_delay=0)

    def test_deduplicates_across_sources(self):
        collector = self._make_collector()
        ordered = [
            # show_hn: top stories
            [1],
            _hn_make_item(1, "https://dup.com", "Show HN: Dup", 100),
            # best: best stories
            [1],
            _hn_make_item(1, "https://dup.com", "Show HN: Dup", 100),
            # top: top stories
            [1],
            _hn_make_item(1, "https://dup.com", "Show HN: Dup", 100),
            # new: new stories
            [1],
            _hn_make_item(1, "https://dup.com", "Show HN: Dup", 100),
        ]
        iterator = iter(ordered)
        with patch.object(collector, "_get_json", side_effect=lambda p: next(iterator)):
            results = collector.collect_all(limit=100)
        assert len(results) == 1
        collector.close()

    def test_respects_limit(self):
        collector = self._make_collector()
        items = [_hn_make_item(i, f"https://{i}.com", f"Story {i}", i) for i in range(1, 21)]
        story_ids = list(range(1, 21))

        ordered = [
            # show_hn
            story_ids,
            *[items[i] for i in range(20)],
            # best
            story_ids,
            *[items[i] for i in range(20)],
            # top
            story_ids,
            *[items[i] for i in range(20)],
            # new
            story_ids,
            *[items[i] for i in range(20)],
        ]
        iterator = iter(ordered)
        with patch.object(collector, "_get_json", side_effect=lambda p: next(iterator)):
            results = collector.collect_all(limit=5)
        assert len(results) <= 5
        collector.close()

    def test_returns_correct_shape(self):
        collector = self._make_collector()
        ordered = [
            [1],
            _hn_make_item(1, "https://a.com", "Title A", 42),
            [1],
            _hn_make_item(1, "https://a.com", "Title A", 42),
            [1],
            _hn_make_item(1, "https://a.com", "Title A", 42),
            [1],
            _hn_make_item(1, "https://a.com", "Title A", 42),
        ]
        iterator = iter(ordered)
        with patch.object(collector, "_get_json", side_effect=lambda p: next(iterator)):
            results = collector.collect_all(limit=10)
        assert len(results) >= 1
        item = results[0]
        assert "url" in item
        assert "source" in item
        assert "source_type" in item
        assert "title" in item
        assert "score" in item
        assert item["source"] == "hackernews"
        collector.close()


class TestHackerNewsRateLimiting:
    """Test rate limit enforcement."""

    def test_rate_limit_sleeps_when_needed(self):
        import time

        collector = HackerNewsCollector(rate_limit_delay=0.5)
        collector._last_request_time = 0.0
        # First call should not sleep
        collector._rate_limit()
        # Second call immediately should sleep
        start = time.monotonic()
        collector._rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.4  # tolerance
        collector.close()

    def test_rate_limit_no_sleep_when_enough_time_passed(self):
        import time

        collector = HackerNewsCollector(rate_limit_delay=0.1)
        collector._last_request_time = 0.0
        # Simulate enough time passing
        collector._last_request_time = time.monotonic() - 1.0
        start = time.monotonic()
        collector._rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05  # should not sleep
        collector.close()


class TestHackerNewsErrorHandling:
    """Test HTTP error handling."""

    def _make_collector(self) -> HackerNewsCollector:
        return HackerNewsCollector(rate_limit_delay=0)

    def test_get_json_handles_connection_error(self):
        collector = self._make_collector()
        with patch.object(collector._session, "get", side_effect=requests.exceptions.ConnectionError):
            result = collector._get_json("/test.json")
        assert result is None
        collector.close()

    def test_get_json_handles_timeout(self):
        collector = self._make_collector()
        with patch.object(collector._session, "get", side_effect=requests.exceptions.Timeout):
            result = collector._get_json("/test.json")
        assert result is None
        collector.close()

    def test_get_json_handles_invalid_json(self):
        collector = self._make_collector()
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("Invalid JSON")
        mock_resp.raise_for_status = MagicMock()
        with patch.object(collector._session, "get", return_value=mock_resp):
            result = collector._get_json("/test.json")
        assert result is None
        collector.close()

    def test_get_json_handles_http_error(self):
        collector = self._make_collector()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        with patch.object(collector._session, "get", return_value=mock_resp):
            result = collector._get_json("/test.json")
        assert result is None
        collector.close()

    def test_fetch_stories_handles_non_numeric_ids(self):
        collector = self._make_collector()
        with patch.object(collector, "_get_json", return_value=[1, "bad", 3]):
            story_ids = collector._fetch_story_ids("new")
        assert story_ids == [1, 3]  # "bad" filtered out
        collector.close()

    def test_fetch_stories_handles_unknown_endpoint(self):
        collector = self._make_collector()
        story_ids = collector._fetch_story_ids("unknown")
        assert story_ids == []
        collector.close()


class TestHackerNewsExtractUrl:
    """Test _extract_url_from_item static method."""

    def test_valid_item(self):
        result = HackerNewsCollector._extract_url_from_item(
            {"url": "https://example.com", "title": "Test", "score": 42}, "top"
        )
        assert result is not None
        assert result["url"] == "https://example.com"
        assert result["source_type"] == "top"
        assert result["title"] == "Test"
        assert result["score"] == 42
        assert result["source"] == "hackernews"

    def test_no_url(self):
        result = HackerNewsCollector._extract_url_from_item(
            {"title": "Ask HN: no url", "score": 10}, "new"
        )
        assert result is None

    def test_empty_url(self):
        result = HackerNewsCollector._extract_url_from_item(
            {"url": "", "title": "Empty", "score": 5}, "best"
        )
        assert result is None

    def test_missing_title_defaults_to_empty(self):
        result = HackerNewsCollector._extract_url_from_item(
            {"url": "https://example.com"}, "top"
        )
        assert result is not None
        assert result["title"] == ""

    def test_missing_score_defaults_to_zero(self):
        result = HackerNewsCollector._extract_url_from_item(
            {"url": "https://example.com", "title": "X"}, "top"
        )
        assert result is not None
        assert result["score"] == 0


class TestHackerNewsContextManager:
    """Test context manager support."""

    def test_with_statement(self):
        with HackerNewsCollector(rate_limit_delay=0) as collector:
            assert isinstance(collector, HackerNewsCollector)

    def test_close(self):
        collector = HackerNewsCollector(rate_limit_delay=0)
        collector.close()  # should not raise


class TestHackerNewsConstants:
    """Test module constants."""

    def test_story_endpoints_defined(self):
        assert "new" in STORY_ENDPOINTS
        assert "top" in STORY_ENDPOINTS
        assert "best" in STORY_ENDPOINTS

    def test_story_endpoints_values(self):
        assert STORY_ENDPOINTS["new"] == "/newstories.json"
        assert STORY_ENDPOINTS["top"] == "/topstories.json"
        assert STORY_ENDPOINTS["best"] == "/beststories.json"


# ======================================================================
# RSS Collector tests
# ======================================================================

from src.collectors.rss import (
    CURATED_FEEDS,
    MAX_REDIRECTS,
    MAX_URLS,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    RSSCollector,
    RSSCollectorError,
)

import feedparser
import re


FAKE_RSS_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Blog</title>
    <link>https://example.com</link>
    <description>A test feed</description>
    <item>
      <title>First Article</title>
      <link>https://other-site.com/article-one</link>
      <description>Great stuff about web dev</description>
    </item>
    <item>
      <title>Second Article</title>
      <link>https://other-site.com/article-two</link>
      <description>More great stuff</description>
    </item>
    <item>
      <title>Third Article</title>
      <link>https://other-site.com/article-three</link>
      <description>Even more stuff</description>
    </item>
  </channel>
</rss>
"""

FAKE_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <link href="https://atom.example.com"/>
  <entry>
    <title>Atom Entry One</title>
    <link href="https://atom.other.com/entry-one"/>
    <summary>An atom entry</summary>
  </entry>
  <entry>
    <title>Atom Entry Two</title>
    <link href="https://atom.other.com/entry-two"/>
    <summary>Another atom entry</summary>
  </entry>
</feed>
"""


def _rss_make_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Create a mock requests.Response for RSS tests."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.RequestException(
            f"HTTP {status_code}"
        )
    return resp


def _rss_make_collector(
    feeds: list[tuple[str, str]],
) -> RSSCollector:
    """Create an RSSCollector pre-loaded with the given feeds."""
    collector = RSSCollector()
    collector._last_request_time = 0.0
    for url, cat in feeds:
        collector.add_feed(url, cat)
    return collector


class TestRSSAddFeed:
    """Tests for RSSCollector.add_feed."""

    def test_add_single_feed(self):
        collector = RSSCollector()
        collector.add_feed("https://example.com/rss", "webdev")
        assert len(collector._feeds) == 1
        assert collector._feeds[0] == {
            "url": "https://example.com/rss",
            "category": "webdev",
        }

    def test_add_multiple_feeds(self):
        collector = RSSCollector()
        collector.add_feed("https://a.com/rss", "indie")
        collector.add_feed("https://b.com/rss", "webdev")
        collector.add_feed("https://c.com/rss", "art")
        assert len(collector._feeds) == 3

    def test_no_duplicates(self):
        collector = RSSCollector()
        collector.add_feed("https://a.com/rss", "webdev")
        collector.add_feed("https://a.com/rss", "webdev")
        assert len(collector._feeds) == 1

    def test_same_url_different_category_skipped(self):
        collector = RSSCollector()
        collector.add_feed("https://a.com/rss", "indie")
        collector.add_feed("https://a.com/rss", "webdev")
        assert len(collector._feeds) == 1
        assert collector._feeds[0]["category"] == "indie"

    def test_empty_url_raises(self):
        collector = RSSCollector()
        with pytest.raises(ValueError, match="Feed URL cannot be empty"):
            collector.add_feed("", "webdev")

    def test_whitespace_url_raises(self):
        collector = RSSCollector()
        with pytest.raises(ValueError, match="Feed URL cannot be empty"):
            collector.add_feed("   ", "webdev")

    def test_empty_category_raises(self):
        collector = RSSCollector()
        with pytest.raises(ValueError, match="Category cannot be empty"):
            collector.add_feed("https://a.com/rss", "")


class TestRSSCuratedFeeds:
    """Tests for the curated feed list."""

    def test_curated_feed_count(self):
        assert 15 <= len(CURATED_FEEDS) <= 20

    def test_all_curated_feeds_are_tuples(self):
        for item in CURATED_FEEDS:
            assert isinstance(item, tuple)
            assert len(item) == 2
            url, category = item
            assert url.startswith("https://")
            assert isinstance(category, str)
            assert len(category) > 0

    def test_curated_categories_are_valid(self):
        valid_categories = {
            "indie", "webdev", "creative", "art",
            "tools", "opensource", "culture",
        }
        for _, category in CURATED_FEEDS:
            assert category in valid_categories, f"Invalid category: {category}"

    def test_load_curated_feeds(self):
        collector = RSSCollector()
        collector.load_curated_feeds()
        assert len(collector._feeds) == len(CURATED_FEEDS)

    def test_no_duplicate_curated_urls(self):
        urls = [url for url, _ in CURATED_FEEDS]
        assert len(urls) == len(set(urls)), "Duplicate URLs in curated feeds"


class TestRSSFetchFeed:
    """Tests for RSSCollector.fetch_feed."""

    def test_fetch_valid_rss(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        mock_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            feed = collector.fetch_feed("https://example.com/rss")
        assert len(feed.entries) == 3
        assert feed.feed.title == "Test Blog"

    def test_fetch_valid_atom(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        mock_resp = _rss_make_response(FAKE_ATOM_XML.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            feed = collector.fetch_feed("https://atom.example.com/feed")
        assert len(feed.entries) == 2

    def test_fetch_timeout_raises(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        with patch.object(
            collector._session, "get",
            side_effect=requests.exceptions.Timeout(),
        ):
            with pytest.raises(RSSCollectorError, match="Timeout"):
                collector.fetch_feed("https://example.com/slow")

    def test_fetch_connection_error_raises(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        with patch.object(
            collector._session, "get",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(RSSCollectorError, match="Connection error"):
                collector.fetch_feed("https://down.example.com/rss")

    def test_fetch_too_many_redirects_raises(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        with patch.object(
            collector._session, "get",
            side_effect=requests.exceptions.TooManyRedirects(),
        ):
            with pytest.raises(RSSCollectorError, match="Too many redirects"):
                collector.fetch_feed("https://redirect.example.com/rss")

    def test_fetch_http_error_raises(self):
        collector = RSSCollector()
        collector._last_request_time = 0.0
        mock_resp = _rss_make_response(b"", status_code=404)
        with patch.object(collector._session, "get", return_value=mock_resp):
            with pytest.raises(RSSCollectorError, match="HTTP"):
                collector.fetch_feed("https://example.com/404")


class TestRSSExtractLinks:
    """Tests for link extraction from feed entries."""

    def test_extract_simple_link(self):
        entry = feedparser.parse(
            '<entry><link href="https://other.com/post"/></entry>'
        ).entries[0]
        links = RSSCollector._extract_links(
            entry, "https://feed.example.com"
        )
        assert "https://other.com/post" in links

    def test_excludes_feed_domain(self):
        entry = feedparser.parse(
            '<entry><link href="https://feed.example.com/page"/></entry>'
        ).entries[0]
        links = RSSCollector._extract_links(
            entry, "https://feed.example.com"
        )
        assert len(links) == 0

    def test_is_valid_url_valid(self):
        assert RSSCollector._is_valid_url("https://example.com") is True
        assert RSSCollector._is_valid_url("http://example.com") is True

    def test_is_valid_url_invalid(self):
        assert RSSCollector._is_valid_url("") is False
        assert RSSCollector._is_valid_url("not-a-url") is False
        assert RSSCollector._is_valid_url("ftp://example.com") is False
        assert RSSCollector._is_valid_url(None) is False


class TestRSSCollectAll:
    """Tests for RSSCollector.collect_all."""

    def test_collect_all_returns_correct_format(self):
        collector = _rss_make_collector(
            [("https://feed-a.com/rss", "webdev")]
        )
        mock_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            results = collector.collect_all(limit=100)

        assert len(results) >= 1
        for item in results:
            assert "url" in item
            assert "source" in item
            assert "source_type" in item
            assert "title" in item
            assert "description" in item
            assert item["source"] == "rss"
            assert item["source_type"] == "webdev"
            assert item["url"].startswith("https://")

    def test_collect_all_respects_limit(self):
        collector = _rss_make_collector(
            [("https://feed-a.com/rss", "webdev")]
        )
        mock_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            results = collector.collect_all(limit=2)
        assert len(results) <= 2

    def test_collect_all_deduplicates_urls(self):
        collector = _rss_make_collector(
            [
                ("https://feed-a.com/rss", "webdev"),
                ("https://feed-b.com/rss", "indie"),
            ]
        )
        mock_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            results = collector.collect_all(limit=100)

        urls = [r["url"] for r in results]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    def test_collect_all_skips_failed_feeds(self):
        """Failed feeds should be skipped without crashing."""
        collector = _rss_make_collector(
            [
                ("https://feed-a.com/rss", "webdev"),
                ("https://feed-b.com/rss", "indie"),
            ]
        )
        good_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))

        def mock_get(url, **kwargs):
            if "feed-a.com" in url:
                return good_resp
            raise requests.exceptions.ConnectionError("refused")

        with patch.object(collector._session, "get", side_effect=mock_get):
            results = collector.collect_all(limit=100)

        assert len(results) >= 1
        assert all(r["source_type"] == "webdev" for r in results)

    def test_collect_all_empty_when_no_feeds(self):
        collector = RSSCollector()
        results = collector.collect_all()
        assert results == []

    def test_collect_all_multiple_feeds(self):
        collector = _rss_make_collector(
            [
                ("https://feed-a.com/rss", "webdev"),
                ("https://atom-feed.com/atom", "creative"),
            ]
        )
        rss_resp = _rss_make_response(FAKE_RSS_XML.encode("utf-8"))
        atom_resp = _rss_make_response(FAKE_ATOM_XML.encode("utf-8"))

        def mock_get(url, **kwargs):
            if "feed-a.com" in url:
                return rss_resp
            return atom_resp

        with patch.object(collector._session, "get", side_effect=mock_get):
            results = collector.collect_all(limit=100)

        categories = {r["source_type"] for r in results}
        assert "webdev" in categories
        assert "creative" in categories

    def test_collect_all_description_truncation(self):
        """Descriptions longer than 500 chars should be truncated."""
        long_desc = "A" * 600
        xml = f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Long Desc Blog</title>
            <link>https://long.example.com</link>
            <item>
              <title>Post</title>
              <link>https://other-site.com/post</link>
              <description>{long_desc}</description>
            </item>
          </channel>
        </rss>
        """
        collector = _rss_make_collector(
            [("https://long.example.com/rss", "webdev")]
        )
        mock_resp = _rss_make_response(xml.encode("utf-8"))
        with patch.object(collector._session, "get", return_value=mock_resp):
            results = collector.collect_all(limit=1)

        assert len(results) == 1
        assert len(results[0]["description"]) <= 500


class TestRSSRateLimiting:
    """Tests for request rate limiting."""

    @patch("src.collectors.rss.time.sleep")
    @patch("src.collectors.rss.time.monotonic")
    def test_rate_limit_enforces_delay(self, mock_mono, mock_sleep):
        collector = RSSCollector()
        mock_mono.return_value = 1.0
        collector._last_request_time = 0.5
        collector._rate_limit()
        mock_sleep.assert_called_once()

    @patch("src.collectors.rss.time.sleep")
    @patch("src.collectors.rss.time.monotonic")
    def test_rate_limit_skips_when_enough_time(self, mock_mono, mock_sleep):
        collector = RSSCollector()
        mock_mono.return_value = 3.0
        collector._last_request_time = 0.5
        collector._rate_limit()
        mock_sleep.assert_not_called()

    @patch("src.collectors.rss.time.sleep")
    @patch("src.collectors.rss.time.monotonic")
    def test_rate_limit_first_request_no_sleep(self, mock_mono, mock_sleep):
        collector = RSSCollector()
        mock_mono.return_value = 1.0
        collector._last_request_time = 0
        collector._rate_limit()
        mock_sleep.assert_not_called()


class TestRSSConstants:
    """Tests for module constants."""

    def test_max_urls_is_100(self):
        assert MAX_URLS == 100

    def test_request_delay_is_reasonable(self):
        assert 1.0 <= REQUEST_DELAY <= 5.0

    def test_max_redirects_is_limited(self):
        assert MAX_REDIRECTS <= 3

    def test_request_timeout_is_reasonable(self):
        assert 5 <= REQUEST_TIMEOUT <= 60
