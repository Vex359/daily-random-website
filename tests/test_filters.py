"""Tests for content filters.

Verifies domain filtering, safety checks, quality scoring,
and deduplication logic.

All network requests are mocked — no real HTTP calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.filters.dedup import URLNormalizer
from src.filters.domains import (
    DomainFilter,
    MAJOR_CORPORATIONS,
    UNSAFE_DOMAINS,
    COMMERCIAL_PREFIXES,
    COMMERCIAL_TLDS,
)
from src.filters.safety import SafetyFilter, MAX_RESPONSE_SIZE, _decode_url_encoding
from src.filters.quality import QualityFilter, MIN_CONTENT_WORDS


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def normalizer(tmp_path: Path) -> URLNormalizer:
    return URLNormalizer(seen_domains_path=tmp_path / "seen.json")


@pytest.fixture()
def normalizer_with_queries(tmp_path: Path) -> URLNormalizer:
    return URLNormalizer(strip_query=False, seen_domains_path=tmp_path / "seen.json")


@pytest.fixture()
def normalizer_with_fragments(tmp_path: Path) -> URLNormalizer:
    return URLNormalizer(strip_fragments=False, seen_domains_path=tmp_path / "seen.json")


@pytest.fixture()
def domain_filter() -> DomainFilter:
    return DomainFilter()


@pytest.fixture()
def safety_filter() -> SafetyFilter:
    return SafetyFilter()


@pytest.fixture()
def quality_filter() -> QualityFilter:
    return QualityFilter()


# =========================================================================
# URLNormalizer tests (existing + new)
# =========================================================================

class TestNormalizeUrl:

    def test_http_to_https(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("http://example.com") == "https://example.com"

    def test_remove_trailing_slash(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/") == "https://example.com"

    def test_remove_trailing_slash_preserves_path(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/page/") == "https://example.com/page"

    def test_remove_index_html(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/index.html") == "https://example.com"

    def test_remove_index_htm(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/index.htm") == "https://example.com"

    def test_remove_index_html_in_subpath(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/dir/index.html") == "https://example.com/dir"

    def test_remove_www_prefix(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://www.example.com") == "https://example.com"

    def test_lowercase_domain(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://EXAMPLE.COM") == "https://example.com"

    def test_remove_default_port_443(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com:443/page") == "https://example.com/page"

    def test_remove_default_port_80_http(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("http://example.com:80/page") == "https://example.com/page"

    def test_remove_default_port_443_on_http(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("http://example.com:443/page") == "https://example.com/page"

    def test_keep_non_default_port(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com:8080/page") == "https://example.com:8080/page"

    def test_remove_fragment(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_strip_query_by_default(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/page?q=1&r=2") == "https://example.com/page"

    def test_keep_query_when_configured(self, normalizer_with_queries: URLNormalizer) -> None:
        assert normalizer_with_queries.normalize_url("https://example.com/page?q=1&r=2") == "https://example.com/page?q=1&r=2"

    def test_keep_fragment_when_configured(self, normalizer_with_fragments: URLNormalizer) -> None:
        assert normalizer_with_fragments.normalize_url("https://example.com/page#section") == "https://example.com/page#section"

    def test_bare_domain_slash(self, normalizer: URLNormalizer) -> None:
        result = normalizer.normalize_url("https://example.com/")
        assert result == "https://example.com"

    def test_combined_normalizations(self, normalizer: URLNormalizer) -> None:
        url = "http://WWW.Example.COM:443/page/INDEX.HTML?ref=home#top"
        result = normalizer.normalize_url(url)
        assert result == "https://example.com/page"

    def test_original_url_not_modified(self, normalizer: URLNormalizer) -> None:
        original = "http://EXAMPLE.COM:80/path/"
        _ = normalizer.normalize_url(original)
        assert original == "http://EXAMPLE.COM:80/path/"


class TestExtractDomain:

    def test_simple_domain(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://example.com") == "example.com"

    def test_subdomain_stripped(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://blog.example.com") == "example.com"

    def test_www_stripped(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://www.example.com") == "example.com"

    def test_multi_part_tld_co_uk(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://example.co.uk") == "example.co.uk"

    def test_subdomain_with_multi_part_tld(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://blog.news.example.co.uk") == "example.co.uk"

    def test_www_with_multi_part_tld(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://www.example.co.uk") == "example.co.uk"

    def test_com_au(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://example.com.au") == "example.com.au"

    def test_bare_domain_string(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("blog.example.com") == "example.com"

    def test_two_part_domain(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://google.com") == "google.com"

    def test_single_label_domain(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("http://localhost") == "localhost"

    def test_case_insensitive(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://EXAMPLE.COM") == "example.com"

    def test_gov_uk(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://www.gov.uk") == "gov.uk"

    def test_ac_uk(self, normalizer: URLNormalizer) -> None:
        assert normalizer.extract_domain("https://www.cam.ac.uk") == "cam.ac.uk"


class TestNormalizeUrls:

    def test_returns_list_of_dicts(self, normalizer: URLNormalizer) -> None:
        results = normalizer.normalize_urls(["https://example.com"])
        assert len(results) == 1
        assert set(results[0].keys()) == {"url", "canonical_url", "domain"}

    def test_preserves_original_url(self, normalizer: URLNormalizer) -> None:
        original = "http://WWW.EXAMPLE.COM:80/page/#top"
        results = normalizer.normalize_urls([original])
        assert results[0]["url"] == original
        assert results[0]["canonical_url"] == "https://example.com/page"

    def test_multiple_urls(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://example.com",
            "http://blog.test.co.uk/post",
            "https://sub.deep.example.org/path",
        ]
        results = normalizer.normalize_urls(urls)
        assert len(results) == 3
        assert results[0]["domain"] == "example.com"
        assert results[1]["domain"] == "test.co.uk"
        assert results[2]["domain"] == "example.org"

    def test_empty_list(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_urls([]) == []


class TestDeduplicate:

    def test_removes_exact_canonical_duplicates(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://alpha.com",
            "http://alpha.com",
            "https://www.beta.com/page",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 2

    def test_removes_domain_duplicates(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://alpha.com/page1",
            "https://alpha.com/page2",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 1
        assert results[0]["domain"] == "alpha.com"

    def test_different_domains_kept(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://alpha.com",
            "https://beta.com",
            "https://gamma.com",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 3

    def test_www_and_non_www_same_domain(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://www.alpha.com/page1",
            "https://alpha.com/page2",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 1

    def test_subdomain_dedup(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://blog.alpha.com/post1",
            "https://news.alpha.com/post2",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 1

    def test_preserves_original_in_output(self, normalizer: URLNormalizer) -> None:
        urls = ["http://ALPHA.com:80/Path/"]
        results = normalizer.deduplicate(urls)
        assert results[0]["url"] == "http://ALPHA.com:80/Path/"
        assert results[0]["canonical_url"] == "https://alpha.com/path"

    def test_seen_domains_filtering(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text(json.dumps(["alpha.com"]), encoding="utf-8")

        norm = URLNormalizer(seen_domains_path=seen_file)
        urls = [
            "https://alpha.com/new-page",
            "https://beta.com/page",
        ]
        results = norm.deduplicate(urls)
        assert len(results) == 1
        assert results[0]["domain"] == "beta.com"

    def test_saves_new_domains(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text(json.dumps([]), encoding="utf-8")

        norm = URLNormalizer(seen_domains_path=seen_file)
        norm.deduplicate(["https://alpha.com", "https://beta.com"])

        saved = json.loads(seen_file.read_text(encoding="utf-8"))
        assert "alpha.com" in saved
        assert "beta.com" in saved

    def test_empty_input(self, normalizer: URLNormalizer) -> None:
        assert normalizer.deduplicate([]) == []

    def test_complex_mixed_duplicates(self, normalizer: URLNormalizer) -> None:
        urls = [
            "https://www.techcrunch.com/article1",
            "http://techcrunch.com/article1",
            "https://news.ycombinator.com/item1",
            "http://news.ycombinator.com/item2",
            "https://blog.arstechnica.co.uk/post",
            "https://www.blog.arstechnica.co.uk/other",
            "https://github.com/user/repo",
            "https://github.com/user/other-repo",
            "https://arxiv.org/abs/2301.00001",
            "https://arxiv.org/abs/2301.00002",
        ]
        results = normalizer.deduplicate(urls)
        assert len(results) == 5
        domains = {r["domain"] for r in results}
        assert domains == {
            "techcrunch.com",
            "ycombinator.com",
            "arstechnica.co.uk",
            "github.com",
            "arxiv.org",
        }


class TestSeenDomainsPersistence:

    def test_load_empty_file(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text("[]", encoding="utf-8")
        norm = URLNormalizer(seen_domains_path=seen_file)
        assert norm.get_seen_domains() == set()

    def test_load_existing_domains(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text(json.dumps(["a.com", "b.org"]), encoding="utf-8")
        norm = URLNormalizer(seen_domains_path=seen_file)
        assert norm.get_seen_domains() == {"a.com", "b.org"}

    def test_missing_file_no_error(self, tmp_path: Path) -> None:
        norm = URLNormalizer(seen_domains_path=tmp_path / "nonexistent.json")
        assert norm.get_seen_domains() == set()

    def test_corrupt_file_no_error(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text("NOT JSON!!!", encoding="utf-8")
        norm = URLNormalizer(seen_domains_path=seen_file)
        assert norm.get_seen_domains() == set()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "subdir" / "seen.json"
        norm = URLNormalizer(seen_domains_path=seen_file)
        norm.deduplicate(["https://example.com"])
        assert seen_file.exists()
        saved = json.loads(seen_file.read_text(encoding="utf-8"))
        assert "example.com" in saved


class TestEdgeCases:

    def test_path_lowercased(self, normalizer: URLNormalizer) -> None:
        assert normalizer.normalize_url("https://example.com/My/Page") == "https://example.com/my/page"

    def test_empty_path(self, normalizer: URLNormalizer) -> None:
        result = normalizer.normalize_url("https://example.com")
        assert result == "https://example.com"

    def test_multiple_trailing_slashes(self, normalizer: URLNormalizer) -> None:
        result = normalizer.normalize_url("https://example.com/page///")
        assert result == "https://example.com/page"

    def test_no_scheme_input_to_normalize(self, normalizer: URLNormalizer) -> None:
        result = normalizer.normalize_url("example.com")
        assert "example.com" in result


# =========================================================================
# DomainFilter tests
# =========================================================================

class TestDomainFilter:

    def test_blocklist_size(self) -> None:
        """Blocklist must contain 50+ domains (acceptance criteria)."""
        total = len(MAJOR_CORPORATIONS) + len(UNSAFE_DOMAINS)
        assert total >= 50, f"Blocklist too small: {total} domains (need 50+)"

    def test_major_corporations_all_blocked(self, domain_filter: DomainFilter) -> None:
        """Every major corporation domain must be blocked."""
        for corp in MAJOR_CORPORATIONS:
            assert domain_filter.is_blocked(corp), f"{corp} should be blocked"

    def test_unsafe_domains_all_blocked(self, domain_filter: DomainFilter) -> None:
        """Every unsafe domain must be blocked."""
        for domain in UNSAFE_DOMAINS:
            assert domain_filter.is_blocked(domain), f"{domain} should be blocked"

    def test_unknown_domain_not_blocked(self, domain_filter: DomainFilter) -> None:
        """An unknown indie domain should not be blocked."""
        assert not domain_filter.is_blocked("random-indie-blog.net")
        assert not domain_filter.is_blocked("cool-personal-site.org")
        assert not domain_filter.is_blocked("awesome-portfolio.dev")

    def test_is_major_corporation_exact(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.is_major_corporation("google.com")
        assert domain_filter.is_major_corporation("facebook.com")
        assert domain_filter.is_major_corporation("amazon.com")
        assert domain_filter.is_major_corporation("apple.com")
        assert domain_filter.is_major_corporation("microsoft.com")

    def test_is_major_corporation_subdomain(self, domain_filter: DomainFilter) -> None:
        """Subdomains of major corps should be detected."""
        assert domain_filter.is_major_corporation("mail.google.com")
        assert domain_filter.is_major_corporation("docs.google.com")
        assert domain_filter.is_major_corporation("www.facebook.com")
        assert domain_filter.is_major_corporation("store.apple.com")

    def test_is_major_corporation_false(self, domain_filter: DomainFilter) -> None:
        assert not domain_filter.is_major_corporation("random-blog.com")
        assert not domain_filter.is_major_corporation("indie-dev.net")

    def test_is_commercial_shop_prefix(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.is_commercial("shop.clothes.com")
        assert domain_filter.is_commercial("store.gadgets.com")
        assert domain_filter.is_commercial("buy.cheap-stuff.com")
        assert domain_filter.is_commercial("price.comparison.com")
        assert domain_filter.is_commercial("deal.hunter.com")

    def test_is_commercial_sale_prefix(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.is_commercial("sale.shop.com")
        assert domain_filter.is_commercial("discount.store.com")
        assert domain_filter.is_commercial("bargain.buy.com")

    def test_is_commercial_tld(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.is_commercial("clothes.shop")
        assert domain_filter.is_commercial("gadgets.store")

    def test_is_commercial_false(self, domain_filter: DomainFilter) -> None:
        assert not domain_filter.is_commercial("random-blog.net")
        assert not domain_filter.is_commercial("tech-news.org")
        assert not domain_filter.is_commercial("my-portfolio.dev")

    def test_get_domain_category_major_corp(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("google.com") == "major_corporation"
        assert domain_filter.get_domain_category("facebook.com") == "major_corporation"

    def test_get_domain_category_unsafe(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("casino.com") == "unsafe"
        assert domain_filter.get_domain_category("gambling.com") == "unsafe"

    def test_get_domain_category_commercial(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("shop.example.com") == "commercial"
        assert domain_filter.get_domain_category("buy.cheap-stuff.com") == "commercial"
        assert domain_filter.get_domain_category("clothes.shop") == "commercial"

    def test_get_domain_category_login(self, domain_filter: DomainFilter) -> None:
        # accounts.google.com resolves to google.com which is major_corp
        assert domain_filter.get_domain_category("accounts.google.com") == "major_corporation"
        assert domain_filter.get_domain_category("login.yahoo.com") == "login"
        assert domain_filter.get_domain_category("auth.example.com") == "login"

    def test_get_domain_category_news(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("daily-news.com") == "news_media"
        assert domain_filter.get_domain_category("tech-times.org") == "news_media"

    def test_get_domain_category_tech(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("cool-code.dev") == "technology"
        assert domain_filter.get_domain_category("byte-hack.io") == "technology"

    def test_get_domain_category_education(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("learn-python.org") == "education"
        assert domain_filter.get_domain_category("university.example.com") == "education"

    def test_get_domain_category_government(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("usa.gov") == "government"
        assert domain_filter.get_domain_category("data.gov") == "government"

    def test_get_domain_category_social(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("community-forum.com") == "social_media"

    def test_get_domain_category_unknown(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.get_domain_category("random-site.com") == "unknown"
        assert domain_filter.get_domain_category("my-cool-page.org") == "unknown"

    def test_extra_blocked_domains(self) -> None:
        """Custom blocked domains can be added at init."""
        df = DomainFilter(extra_blocked={"my-custom-block.com"})
        assert df.is_blocked("my-custom-block.com")
        assert not df.is_blocked("random-other.com")

    def test_extra_corporations(self) -> None:
        """Custom corporation domains can be added at init."""
        df = DomainFilter(extra_corporations={"mega-corp.com"})
        assert df.is_major_corporation("mega-corp.com")

    def test_extra_unsafe(self) -> None:
        """Custom unsafe domains can be added at init."""
        df = DomainFilter(extra_unsafe={"custom-bad.com"})
        assert df.is_blocked("custom-bad.com")

    def test_filter_urls_removes_blocked(self, domain_filter: DomainFilter) -> None:
        urls = [
            "https://google.com/search",
            "https://random-blog.net/post",
            "https://facebook.com/page",
        ]
        results = domain_filter.filter_urls(urls)
        assert len(results) == 1
        assert results[0]["domain"] == "random-blog.net"

    def test_filter_urls_returns_category(self, domain_filter: DomainFilter) -> None:
        urls = ["https://random-blog.com/post"]
        results = domain_filter.filter_urls(urls)
        assert results[0]["category"] == "unknown"

    def test_filter_urls_empty(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.filter_urls([]) == []

    def test_filter_urls_all_blocked(self, domain_filter: DomainFilter) -> None:
        urls = [
            "https://google.com/search",
            "https://facebook.com/page",
        ]
        results = domain_filter.filter_urls(urls)
        assert len(results) == 0

    def test_login_domain_blocked(self, domain_filter: DomainFilter) -> None:
        assert domain_filter.is_blocked("accounts.google.com")
        assert domain_filter.is_blocked("login.yahoo.com")
        assert domain_filter.is_blocked("auth.example.com")
        assert domain_filter.is_blocked("signin.github.com")
        assert domain_filter.is_blocked("sso.example.com")


# =========================================================================
# SafetyFilter tests
# =========================================================================

class TestSafetyFilter:

    def test_safe_url_passes(self, safety_filter: SafetyFilter) -> None:
        assert safety_filter.is_safe_url("https://random-blog.com/2024/my-post")

    def test_safe_url_with_query(self, safety_filter: SafetyFilter) -> None:
        assert safety_filter.is_safe_url("https://example.com/page?name=value")

    def test_blocked_domain_fails(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://google.com/search")
        assert not safety_filter.is_safe_url("https://facebook.com/page")

    def test_javascript_scheme_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("javascript:alert(1)")

    def test_data_scheme_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("data:text/html,<script>alert(1)</script>")

    def test_suspicious_param_redirect(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/page?redirect=http://evil.com")

    def test_suspicious_param_cmd(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/page?cmd=ls")

    def test_suspicious_param_eval(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/page?eval=code")

    def test_suspicious_param_exec(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/page?exec=command")

    def test_suspicious_param_base64(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/page?base64=maliciouspayload")

    def test_path_traversal_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/path/../../../etc/passwd")

    def test_env_file_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/.env")

    def test_git_dir_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/.git/config")

    def test_phpmyadmin_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/phpmyadmin/")

    def test_cgi_bin_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("https://example.com/cgi-bin/script")

    def test_no_hostname_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("http://")

    def test_ftp_scheme_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("ftp://example.com/file")

    def test_empty_url_blocked(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.is_safe_url("")

    def test_safe_content_clean_html(self, safety_filter: SafetyFilter) -> None:
        html = "<html><body><h1>Hello World</h1><p>Content here.</p></body></html>"
        assert safety_filter.is_safe_content(html)

    def test_safe_content_empty(self, safety_filter: SafetyFilter) -> None:
        assert safety_filter.is_safe_content("")

    def test_malware_eval_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<html><script>eval(document.cookie)</script></html>'
        assert not safety_filter.is_safe_content(html)

    def test_malware_hidden_iframe_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<html><iframe src="evil.com" hidden></iframe></html>'
        assert not safety_filter.is_safe_content(html)

    def test_malware_javascript_location_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<script>window.location="javascript:alert(1)"</script>'
        assert not safety_filter.is_safe_content(html)

    def test_malware_crypto_miner_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<script src="https://coinhive.com/lib/miner.js"></script>'
        assert not safety_filter.is_safe_content(html)

    def test_malware_coinhive_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<script>new CoinHive.Anonymous("x")</script>'
        assert not safety_filter.is_safe_content(html)

    def test_malware_cryptoloot_detected(self, safety_filter: SafetyFilter) -> None:
        html = '<script src="https://cryptoloot.pro/lib/miner.js"></script>'
        assert not safety_filter.is_safe_content(html)

    def test_check_response_ok(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.content = b"<html><body>OK</body></html>"
        mock_resp.history = []

        result = safety_filter.check_response(mock_resp)
        assert result["is_safe"]
        assert result["content_type"] == "text/html"
        assert result["size_bytes"] == len(mock_resp.content)
        assert result["issues"] == []

    def test_check_response_non_html(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.content = b'{"key": "value"}'
        mock_resp.history = []

        result = safety_filter.check_response(mock_resp)
        assert not result["is_safe"]
        assert any("Not HTML" in issue for issue in result["issues"])

    def test_check_response_too_large(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b"x" * (MAX_RESPONSE_SIZE + 1)
        mock_resp.history = []

        result = safety_filter.check_response(mock_resp)
        assert not result["is_safe"]
        assert any("too large" in issue for issue in result["issues"])

    def test_check_response_error_status(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b"<html><body>Not Found</body></html>"
        mock_resp.history = []

        result = safety_filter.check_response(mock_resp)
        assert not result["is_safe"]
        assert any("404" in issue for issue in result["issues"])

    def test_check_response_too_many_redirects(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b"<html>OK</html>"
        mock_resp.history = [MagicMock() for _ in range(5)]  # 5 redirects

        result = safety_filter.check_response(mock_resp)
        assert not result["is_safe"]
        assert any("redirects" in issue for issue in result["issues"])

    def test_check_response_malware_content(self, safety_filter: SafetyFilter) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = b'<html><script>eval(document.cookie)</script></html>'
        mock_resp.history = []

        result = safety_filter.check_response(mock_resp)
        assert not result["is_safe"]
        assert any("Unsafe content" in issue for issue in result["issues"])

    def test_has_login_wall_detected(self, safety_filter: SafetyFilter) -> None:
        html = """
        <html>
        <head><title>Sign In</title></head>
        <body>
        <form action="/login">
            <input type="text" name="username">
            <input type="password" name="password">
            <button type="submit">Sign In</button>
        </form>
        <a href="/forgot-password">Forgot password?</a>
        </body>
        </html>
        """
        assert safety_filter.has_login_wall(html)

    def test_has_login_wall_not_detected(self, safety_filter: SafetyFilter) -> None:
        html = """
        <html>
        <head><title>My Blog Post</title></head>
        <body>
        <h1>A Great Article</h1>
        <p>This is some meaningful content about a topic.</p>
        </body>
        </html>
        """
        assert not safety_filter.has_login_wall(html)

    def test_has_login_wall_empty(self, safety_filter: SafetyFilter) -> None:
        assert not safety_filter.has_login_wall("")

    def test_has_login_wall_subscribe(self, safety_filter: SafetyFilter) -> None:
        html = "<html><body><p>Subscribe to continue reading this article.</p></body></html>"
        assert safety_filter.has_login_wall(html)


# =========================================================================
# QualityFilter tests
# =========================================================================

class TestQualityFilter:

    def test_is_valid_url_good(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_valid_url("https://example.com")
        assert quality_filter.is_valid_url("https://blog.example.com/post")
        assert quality_filter.is_valid_url("http://example.com:8080/page")

    def test_is_valid_url_no_scheme(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("example.com")

    def test_is_valid_url_no_hostname(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("https://")

    def test_is_valid_url_short_hostname(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("https://a.b")

    def test_is_valid_url_double_dots(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("https://a..b.com")

    def test_is_valid_url_spaces(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("https://exam ple.com")

    def test_is_valid_url_empty(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("")

    def test_is_valid_url_ftp(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_valid_url("ftp://example.com/file")

    def test_has_content_with_text(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>My Post</title></head>
        <body>
        <h1>Interesting Article</h1>
        <p>This is a meaningful article with enough content to pass
        the quality filter. It contains multiple sentences and provides
        real value to readers who are interested in the topic.</p>
        </body>
        </html>
        """
        assert quality_filter.has_content(html)

    def test_has_content_empty(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.has_content("")

    def test_has_content_too_short(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.has_content("<html><body>Hi</body></html>")

    def test_has_content_404_page(self, quality_filter: QualityFilter) -> None:
        html = "<html><head><title>404 Not Found</title></head><body><h1>404</h1></body></html>"
        assert not quality_filter.has_content(html)

    def test_has_content_under_construction(self, quality_filter: QualityFilter) -> None:
        html = """
        <html><body>
        <h1>Coming Soon</h1>
        <p>This page is under construction. Check back later.</p>
        </body></html>
        """
        assert not quality_filter.has_content(html)

    def test_has_content_error_page(self, quality_filter: QualityFilter) -> None:
        html = """
        <html><body>
        <title>Error</title>
        <h1>Error</h1>
        <p>An error occurred.</p>
        </body></html>
        """
        assert not quality_filter.has_content(html)

    def test_is_too_short_empty(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_too_short("")

    def test_is_too_short_none(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_too_short("")

    def test_is_too_short_just_enough(self, quality_filter: QualityFilter) -> None:
        text = "word " * MIN_CONTENT_WORDS
        assert not quality_filter.is_too_short(text.strip())

    def test_is_too_short_under_limit(self, quality_filter: QualityFilter) -> None:
        text = "one two three"
        assert quality_filter.is_too_short(text)

    def test_is_too_short_custom_min(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_too_short("one two", min_words=5)
        assert not quality_filter.is_too_short("one two three", min_words=3)

    def test_is_login_page_detected(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>Sign In</title></head>
        <body>
        <form>
            <input type="text" name="username" placeholder="Email address">
            <input type="password" name="password">
            <button type="submit">Log In</button>
        </form>
        <a href="/forgot-password">Forgot password?</a>
        <a href="/signup">Create an account</a>
        </body>
        </html>
        """
        assert quality_filter.is_login_page(html)

    def test_is_login_page_not_detected(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>Blog Post</title></head>
        <body>
        <h1>Interesting Article</h1>
        <p>Great content here with many words about various topics.
        This is definitely not a login page but rather a real article
        that readers will find informative and engaging.</p>
        </body>
        </html>
        """
        assert not quality_filter.is_login_page(html)

    def test_is_login_page_empty(self, quality_filter: QualityFilter) -> None:
        assert not quality_filter.is_login_page("")

    def test_is_commercial_title_shop(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_commercial_site("", title="Buy Cheap Gadgets Online")

    def test_is_commercial_title_store(self, quality_filter: QualityFilter) -> None:
        assert quality_filter.is_commercial_site("", title="My Online Store")

    def test_is_commercial_html_cart(self, quality_filter: QualityFilter) -> None:
        html = '<html><body><button>Add to Cart</button><p>Price: $29.99</p></body></html>'
        assert quality_filter.is_commercial_site(html)

    def test_is_commercial_html_buy_now(self, quality_filter: QualityFilter) -> None:
        html = '<html><body><a href="/checkout">Buy Now</a></body></html>'
        assert quality_filter.is_commercial_site(html)

    def test_is_commercial_false(self, quality_filter: QualityFilter) -> None:
        html = '<html><body><h1>Blog Post</h1><p>Content here.</p></body></html>'
        assert not quality_filter.is_commercial_site(html, title="My Blog Post")

    def test_get_page_title_found(self, quality_filter: QualityFilter) -> None:
        html = "<html><head><title>My Page Title</title></head></html>"
        assert quality_filter.get_page_title(html) == "My Page Title"

    def test_get_page_title_not_found(self, quality_filter: QualityFilter) -> None:
        html = "<html><head></head></html>"
        assert quality_filter.get_page_title(html) is None

    def test_get_page_title_too_short(self, quality_filter: QualityFilter) -> None:
        html = "<html><head><title>Hi</title></head></html>"
        assert quality_filter.get_page_title(html) is None

    def test_get_page_title_case_insensitive(self, quality_filter: QualityFilter) -> None:
        html = '<HTML><HEAD><TITLE>Case Insensitive</TITLE></HEAD></HTML>'
        assert quality_filter.get_page_title(html) == "Case Insensitive"

    def test_get_page_title_with_whitespace(self, quality_filter: QualityFilter) -> None:
        html = "<html><head><title>  Spaced Title  </title></head></html>"
        assert quality_filter.get_page_title(html) == "Spaced Title"

    def test_validate_page_good(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>Great Article About Python</title></head>
        <body>
        <h1>Python Tips and Tricks</h1>
        <p>Here are some useful Python tips for developers.
        Learn about list comprehensions, generators, decorators,
        and other powerful Python features that will improve your code.</p>
        </body>
        </html>
        """
        result = quality_filter.validate_page("https://blog.example.com/python-tips", html)
        assert result["url_valid"]
        assert result["has_content"]
        assert not result["is_login"]
        assert not result["is_commercial"]
        assert result["title"] == "Great Article About Python"

    def test_validate_page_invalid_url(self, quality_filter: QualityFilter) -> None:
        result = quality_filter.validate_page("not-a-url", "<html>content</html>")
        assert not result["url_valid"]
        assert "Invalid URL" in result["issues"][0]

    def test_validate_page_login(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>Sign In</title></head>
        <body>
        <form>
            <input type="text" name="username">
            <input type="password" name="password">
            <button>Log In</button>
        </form>
        <a href="/forgot-password">Forgot password?</a>
        <a href="/signup">Create account</a>
        </body>
        </html>
        """
        result = quality_filter.validate_page("https://example.com/login", html)
        assert result["is_login"]

    def test_validate_page_commercial(self, quality_filter: QualityFilter) -> None:
        html = """
        <html>
        <head><title>Shop Online - Buy Now!</title></head>
        <body>
        <h1>Great Deals</h1>
        <p>Buy now and save! Add to cart for the best prices.</p>
        </body>
        </html>
        """
        result = quality_filter.validate_page("https://shop.example.com", html)
        assert result["is_commercial"]

    def test_validate_page_empty_html(self, quality_filter: QualityFilter) -> None:
        result = quality_filter.validate_page("https://example.com", "")
        assert result["url_valid"]
        assert not result["has_content"]
        assert result["issues"]

    def test_validate_page_no_html(self, quality_filter: QualityFilter) -> None:
        """URL-only validation (no HTML fetched)."""
        result = quality_filter.validate_page("https://example.com")
        assert result["url_valid"]
        assert not result["is_login"]
        assert not result["is_commercial"]


# =========================================================================
# Integration: DomainFilter + SafetyFilter + QualityFilter
# =========================================================================

class TestFilterIntegration:

    def test_full_pipeline_flow(self, domain_filter: DomainFilter, safety_filter: SafetyFilter, quality_filter: QualityFilter) -> None:
        """Test a typical URL through all three filters."""
        url = "https://random-blog.com/2024/interesting-post"

        # Step 1: Domain check
        assert not domain_filter.is_blocked(url)

        # Step 2: Safety check
        assert safety_filter.is_safe_url(url)

        # Step 3: Quality check
        assert quality_filter.is_valid_url(url)

    def test_blocked_corp_fails_early(self, domain_filter: DomainFilter, safety_filter: SafetyFilter) -> None:
        """A major corp domain should fail at domain filter."""
        url = "https://google.com/search"
        assert domain_filter.is_blocked(url)
        # Safety filter also catches it since it uses DomainFilter
        assert not safety_filter.is_safe_url(url)

    def test_commercial_fails_domain_check(self, domain_filter: DomainFilter) -> None:
        """Commercial domains should be caught by domain filter."""
        url = "https://shop.example.com/buy-now"
        assert domain_filter.is_blocked(url)

    def test_unsafe_domain_fails_all(self, domain_filter: DomainFilter, safety_filter: SafetyFilter, quality_filter: QualityFilter) -> None:
        """An unsafe domain should fail all safety checks."""
        url = "https://casino.com/slots"
        assert domain_filter.is_blocked(url)
        assert not safety_filter.is_safe_url(url)

    def test_good_url_passes_all(self, domain_filter: DomainFilter, safety_filter: SafetyFilter, quality_filter: QualityFilter) -> None:
        """A good indie URL should pass all checks."""
        url = "https://cool-indie-blog.net/2024/my-post"
        assert not domain_filter.is_blocked(url)
        assert safety_filter.is_safe_url(url)
        assert quality_filter.is_valid_url(url)

    def test_suspicious_url_caught_by_safety(self, domain_filter: DomainFilter, safety_filter: SafetyFilter) -> None:
        """A URL with suspicious params should be caught by safety filter."""
        url = "https://random-blog.com/page?eval=malicious"
        assert not domain_filter.is_blocked(url)  # Domain is fine
        assert not safety_filter.is_safe_url(url)  # But URL is suspicious
