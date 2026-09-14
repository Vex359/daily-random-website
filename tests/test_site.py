"""Tests for static site generator.

Verifies that the site generator produces valid HTML,
includes all featured content, and is deployment-ready.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.site.generator import (
    VALID_CATEGORIES,
    CATEGORY_SLUGS,
    SiteGenerator,
    _render_filter_buttons,
    _render_post_card,
    _generate_index_page,
    _generate_archive_page,
    _generate_category_page,
    _generate_post_page,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_post(
    *,
    title: str = "Test Site",
    url: str = "https://example.com",
    domain: str = "example.com",
    category: str = "Interactive",
    source: str = "hackernews",
    description: str = "A test description.",
    ai_description: str = "An AI-generated Instagram-style description.",
    why_interesting: str = "Because it is interesting.",
    discovered_at: str = "2025-01-15T12:00:00+00:00",
    screenshot_path: str = "screenshots/example-com.webp",
) -> dict:
    """Create a minimal valid post dict for testing."""
    return {
        "id": "test-id-1",
        "url": url,
        "domain": domain,
        "title": title,
        "description": description,
        "ai_description": ai_description,
        "category": category,
        "source": source,
        "score": 80,
        "screenshot_path": screenshot_path,
        "why_interesting": why_interesting,
        "discovered_at": discovered_at,
    }


@pytest.fixture
def sample_posts():
    """Multiple posts across different categories and dates."""
    return [
        _make_post(
            title="Latest Site",
            url="https://latest.com",
            domain="latest.com",
            category="Interactive",
            source="hackernews",
            ai_description="The latest discovery.",
            discovered_at="2025-06-15T10:00:00+00:00",
        ),
        _make_post(
            title="Older Site",
            url="https://older.com",
            domain="older.com",
            category="Weird",
            source="rss",
            ai_description="An older weird site.",
            discovered_at="2025-06-10T10:00:00+00:00",
        ),
        _make_post(
            title="Another Interactive",
            url="https://another.com",
            domain="another.com",
            category="Interactive",
            source="github",
            ai_description="Another interactive site.",
            discovered_at="2025-06-14T10:00:00+00:00",
        ),
        _make_post(
            title="Game Post",
            url="https://game.com",
            domain="game.com",
            category="Games",
            source="hackernews",
            ai_description="A cool game.",
            discovered_at="2025-06-13T10:00:00+00:00",
        ),
    ]


@pytest.fixture
def single_post():
    """Single post for focused tests."""
    return _make_post()


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path / "site_output"


@pytest.fixture
def generator(tmp_output):
    """SiteGenerator with temp output dir."""
    return SiteGenerator(output_dir=tmp_output)


# ---------------------------------------------------------------------------
# Post card rendering
# ---------------------------------------------------------------------------

class TestPostCardRendering:
    """Test individual card HTML output."""

    def test_card_contains_article_tag(self, single_post):
        result = _render_post_card(single_post)
        assert "<article" in result
        assert 'class="post-card"' in result

    def test_card_has_data_category(self, single_post):
        result = _render_post_card(single_post)
        assert 'data-category="interactive"' in result

    def test_card_has_image(self, single_post):
        result = _render_post_card(single_post)
        assert "<img" in result
        assert 'loading="lazy"' in result
        assert "screenshots/example-com.webp" in result

    def test_card_has_title(self, single_post):
        result = _render_post_card(single_post)
        assert 'class="post-title"' in result
        assert "Test Site" in result

    def test_card_has_description(self, single_post):
        result = _render_post_card(single_post)
        assert 'class="post-description"' in result
        assert "AI-generated Instagram-style description" in result

    def test_card_has_category_badge(self, single_post):
        result = _render_post_card(single_post)
        assert 'class="post-category"' in result
        assert "Interactive" in result

    def test_card_has_source(self, single_post):
        result = _render_post_card(single_post)
        assert 'class="post-source"' in result
        assert "via Hacker News" in result

    def test_card_has_visit_button(self, single_post):
        result = _render_post_card(single_post)
        assert 'class="visit-btn"' in result
        assert 'target="_blank"' in result
        assert 'rel="noopener"' in result

    def test_card_escapes_html_in_title(self):
        post = _make_post(title="<script>alert('xss')</script>")
        result = _render_post_card(post)
        assert "&lt;script&gt;" in result

    def test_card_escapes_html_in_description(self):
        post = _make_post(ai_description="<img onerror='alert(1)'>")
        result = _render_post_card(post)
        assert "&lt;img" in result

    def test_card_fallback_to_description(self):
        """When ai_description is empty, falls back to description."""
        post = _make_post(ai_description="", description="Fallback text here")
        result = _render_post_card(post)
        assert "Fallback text here" in result

    def test_source_label_mapping(self):
        """Various source values get correct labels."""
        sources = {
            "hackernews": "Hacker News",
            "github": "GitHub",
            "rss": "RSS Feed",
            "wikipedia": "Wikipedia",
            "awesome_lists": "Awesome Lists",
            "unknown_source": "Unknown Source",
        }
        for source_key, label in sources.items():
            post = _make_post(source=source_key)
            result = _render_post_card(post)
            assert f"via {label}" in result


# ---------------------------------------------------------------------------
# Filter buttons
# ---------------------------------------------------------------------------

class TestFilterButtons:
    """Test filter button rendering."""

    def test_filter_buttons_contain_all_categories(self):
        result = _render_filter_buttons()
        for cat in VALID_CATEGORIES:
            assert cat in result

    def test_filter_buttons_have_all_button(self):
        result = _render_filter_buttons()
        assert 'data-filter="all"' in result
        assert "All" in result

    def test_active_button_marked(self):
        result = _render_filter_buttons("weird")
        assert re.search(r'class="filter-btn active".*data-filter="weird"', result)

    def test_all_others_not_active(self):
        result = _render_filter_buttons("weird")
        active_count = len(re.findall(r'filter-btn active', result))
        assert active_count == 1


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

class TestIndexPage:
    """Test homepage generation."""

    def test_index_has_doctype(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert result.startswith("<!DOCTYPE html>")

    def test_index_has_meta_viewport(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'name="viewport"' in result
        assert 'content="width=device-width, initial-scale=1.0"' in result

    def test_index_has_site_name(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert "The Daily Web" in result

    def test_index_has_semantic_header(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert "<header" in result
        assert 'class="site-header"' in result

    def test_index_has_semantic_main(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert "<main" in result

    def test_index_has_semantic_footer(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert "<footer" in result
        assert 'class="site-footer"' in result

    def test_index_has_hero_section(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'class="hero-section"' in result
        assert "Today" in result

    def test_index_hero_is_latest_post(self, sample_posts):
        """Hero should show the most recent post (Latest Site)."""
        result = _generate_index_page(sample_posts)
        assert "Latest Site" in result

    def test_index_has_recent_grid(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'class="recent-section"' in result
        assert 'class="posts-grid"' in result

    def test_index_recent_shows_last_10(self):
        """Grid should show up to 10 recent posts."""
        posts = [_make_post(
            title=f"Post {i}",
            url=f"https://site{i}.com",
            domain=f"site{i}.com",
            discovered_at=f"2025-06-{15 - i:02d}T10:00:00+00:00",
        ) for i in range(15)]
        result = _generate_index_page(posts)
        # Hero card + 10 recent cards = 11
        card_count = result.count('class="post-card"')
        assert card_count == 11

    def test_index_links_css(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'href="style.css"' in result

    def test_index_links_js(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'src="app.js"' in result

    def test_index_empty_posts(self):
        """Index with no posts should still produce valid HTML."""
        result = _generate_index_page([])
        assert "<!DOCTYPE html>" in result
        assert "The Daily Web" in result
        assert "No discoveries yet" in result

    def test_index_nav_links(self, sample_posts):
        result = _generate_index_page(sample_posts)
        assert 'href="index.html"' in result
        assert 'href="archive.html"' in result

    def test_index_hero_before_recent(self, sample_posts):
        """Hero should appear before recent grid."""
        result = _generate_index_page(sample_posts)
        hero_pos = result.find("hero-section")
        recent_pos = result.find("recent-section")
        assert hero_pos < recent_pos


# ---------------------------------------------------------------------------
# Archive page
# ---------------------------------------------------------------------------

class TestArchivePage:
    """Test archive page generation."""

    def test_archive_has_doctype(self, sample_posts):
        result = _generate_archive_page(sample_posts)
        assert result.startswith("<!DOCTYPE html>")

    def test_archive_has_filter_bar(self, sample_posts):
        result = _generate_archive_page(sample_posts)
        assert 'class="filter-bar"' in result
        assert 'class="filter-buttons"' in result

    def test_archive_has_all_filter_buttons(self, sample_posts):
        result = _generate_archive_page(sample_posts)
        for cat in VALID_CATEGORIES:
            assert cat in result

    def test_archive_has_all_posts(self, sample_posts):
        """All posts should appear in archive regardless of category."""
        result = _generate_archive_page(sample_posts)
        for post in sample_posts:
            assert post["title"] in result

    def test_archive_sorted_newest_first(self, sample_posts):
        """Newest post should appear first in the grid."""
        result = _generate_archive_page(sample_posts)
        latest_pos = result.find("Latest Site")
        older_pos = result.find("Older Site")
        assert latest_pos < older_pos

    def test_archive_empty_posts(self):
        result = _generate_archive_page([])
        assert "No posts yet" in result

    def test_archive_has_semantic_elements(self, sample_posts):
        result = _generate_archive_page(sample_posts)
        assert "<header" in result
        assert "<main" in result
        assert "<footer" in result

    def test_archive_cards_have_data_category(self, sample_posts):
        result = _generate_archive_page(sample_posts)
        assert 'data-category="interactive"' in result
        assert 'data-category="weird"' in result
        assert 'data-category="games"' in result


# ---------------------------------------------------------------------------
# Category page
# ---------------------------------------------------------------------------

class TestCategoryPage:
    """Test category-filtered page generation."""

    def test_category_page_filters_posts(self, sample_posts):
        """Only interactive posts should appear on the interactive page."""
        interactive_posts = [p for p in sample_posts if p["category"] == "Interactive"]
        result = _generate_category_page("Interactive", interactive_posts, sample_posts)
        assert "Latest Site" in result
        assert "Another Interactive" in result
        assert "Game Post" not in result

    def test_category_page_empty_when_no_posts(self):
        result = _generate_category_page("Games", [], [])
        assert "No Games posts yet" in result

    def test_category_page_has_filter_buttons(self, sample_posts):
        interactive_posts = [p for p in sample_posts if p["category"] == "Interactive"]
        result = _generate_category_page("Interactive", interactive_posts, sample_posts)
        assert 'class="filter-buttons"' in result

    def test_category_page_active_button(self, sample_posts):
        weird_posts = [p for p in sample_posts if p["category"] == "Weird"]
        result = _generate_category_page("Weird", weird_posts, sample_posts)
        assert re.search(r'filter-btn active.*data-filter="weird"', result)

    def test_category_page_has_category_in_header(self, sample_posts):
        interactive_posts = [p for p in sample_posts if p["category"] == "Interactive"]
        result = _generate_category_page("Interactive", interactive_posts, sample_posts)
        assert "Interactive" in result

    def test_all_categories_have_pages(self, sample_posts):
        """Each valid category should produce a page without errors."""
        for cat in VALID_CATEGORIES:
            cat_posts = [p for p in sample_posts if p["category"] == cat]
            result = _generate_category_page(cat, cat_posts, sample_posts)
            assert "<!DOCTYPE html>" in result
            assert cat in result

    def test_category_page_title_contains_category(self):
        result = _generate_category_page("Art", [], [])
        assert "Art" in result
        assert "The Daily Web" in result


# ---------------------------------------------------------------------------
# Post detail page
# ---------------------------------------------------------------------------

class TestPostDetailPage:
    """Test individual post page generation."""

    def test_post_page_has_doctype(self, single_post):
        result = _generate_post_page(single_post)
        assert result.startswith("<!DOCTYPE html>")

    def test_post_page_has_post_title(self, single_post):
        result = _generate_post_page(single_post)
        assert 'class="post-detail-title"' in result
        assert "Test Site" in result

    def test_post_page_has_category(self, single_post):
        result = _generate_post_page(single_post)
        assert "Interactive" in result

    def test_post_page_has_source(self, single_post):
        result = _generate_post_page(single_post)
        assert "via Hacker News" in result

    def test_post_page_has_image(self, single_post):
        result = _generate_post_page(single_post)
        assert "screenshots/example-com.webp" in result

    def test_post_page_has_description(self, single_post):
        result = _generate_post_page(single_post)
        assert "AI-generated Instagram-style description" in result

    def test_post_page_has_why_interesting(self, single_post):
        result = _generate_post_page(single_post)
        assert "Because it is interesting" in result

    def test_post_page_has_visit_button(self, single_post):
        result = _generate_post_page(single_post)
        assert 'class="visit-btn visit-btn--large"' in result
        assert 'href="https://example.com"' in result
        assert 'target="_blank"' in result

    def test_post_page_links_css(self, single_post):
        result = _generate_post_page(single_post)
        assert 'href="style.css"' in result

    def test_post_page_links_js(self, single_post):
        result = _generate_post_page(single_post)
        assert 'src="app.js"' in result

    def test_post_page_escapes_html(self):
        """XSS in post fields should be escaped."""
        post = _make_post(
            title="<script>alert('xss')</script>",
            ai_description="<img onerror='steal(cookie)'>",
        )
        result = _generate_post_page(post)
        assert "<script>" not in result
        assert "<img onerror" not in result

    def test_post_page_without_why_interesting(self):
        """Post with empty why_interesting should not show the section."""
        post = _make_post(why_interesting="")
        result = _generate_post_page(post)
        assert "Why interesting" not in result


# ---------------------------------------------------------------------------
# SiteGenerator class
# ---------------------------------------------------------------------------

class TestSiteGenerator:
    """Test the main SiteGenerator class."""

    def test_generate_site_creates_output_dir(self, tmp_path):
        out = tmp_path / "new_dir"
        gen = SiteGenerator(output_dir=out)
        gen.generate_site([])
        assert out.exists()

    def test_generate_site_creates_index(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        assert (generator._output_dir / "index.html").exists()

    def test_generate_site_creates_archive(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        assert (generator._output_dir / "archive.html").exists()

    def test_generate_site_creates_css(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        assert (generator._output_dir / "style.css").exists()

    def test_generate_site_creates_js(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        assert (generator._output_dir / "app.js").exists()

    def test_generate_site_creates_category_pages(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        for cat in VALID_CATEGORIES:
            slug = CATEGORY_SLUGS[cat]
            assert (generator._output_dir / f"{slug}.html").exists()

    def test_generate_site_creates_post_pages(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        post_pages = list(generator._output_dir.glob("post_*.html"))
        assert len(post_pages) == len(sample_posts)

    def test_generate_site_returns_path(self, generator, sample_posts):
        result = generator.generate_site(sample_posts)
        assert result == generator._output_dir

    def test_generate_site_empty_posts(self, generator):
        """Generating with no posts should not crash."""
        generator.generate_site([])
        assert (generator._output_dir / "index.html").exists()
        assert (generator._output_dir / "archive.html").exists()

    def test_generate_site_custom_output_dir(self, tmp_path):
        """Can override output dir in generate_site call."""
        gen = SiteGenerator()
        custom = tmp_path / "custom_output"
        result = gen.generate_site([], output_dir=custom)
        assert result == custom
        assert (custom / "index.html").exists()

    def test_generate_index_returns_string(self, generator, sample_posts):
        result = generator.generate_index(sample_posts)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_archive_returns_string(self, generator, sample_posts):
        result = generator.generate_archive(sample_posts)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_category_returns_string(self, generator, sample_posts):
        result = generator.generate_category("Interactive", sample_posts, sample_posts)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_generate_post_page_returns_string(self, generator, single_post):
        result = generator.generate_post_page(single_post)
        assert isinstance(result, str)
        assert "<!DOCTYPE html>" in result

    def test_css_is_valid_no_external_deps(self, generator, sample_posts):
        """CSS should not reference external fonts or CDNs."""
        generator.generate_site(sample_posts)
        css = (generator._output_dir / "style.css").read_text(encoding="utf-8")
        assert "googleapis.com" not in css
        assert "fonts.googleapis" not in css
        assert "@import url" not in css

    def test_js_is_self_contained(self, generator, sample_posts):
        """JS should not load external scripts."""
        generator.generate_site(sample_posts)
        js_content = (generator._output_dir / "app.js").read_text(encoding="utf-8")
        assert "<script src=" not in js_content
        assert "fetch(" not in js_content

    def test_html_is_valid_utf8(self, generator, sample_posts):
        """All HTML files should start with proper encoding."""
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert 'charset="UTF-8"' in content


# ---------------------------------------------------------------------------
# Responsive design
# ---------------------------------------------------------------------------

class TestResponsiveDesign:
    """Test responsive meta tags and CSS."""

    def test_all_pages_have_viewport_meta(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert 'name="viewport"' in content, f"{html_file.name} missing viewport"

    def test_viewport_is_responsive(self, generator, sample_posts):
        """Viewport should allow scaling."""
        generator.generate_site(sample_posts)
        index = (generator._output_dir / "index.html").read_text(encoding="utf-8")
        assert "width=device-width" in index

    def test_css_has_media_queries(self, generator, sample_posts):
        """CSS should have media queries for mobile."""
        generator.generate_site(sample_posts)
        css = (generator._output_dir / "style.css").read_text(encoding="utf-8")
        assert "@media" in css
        assert "max-width" in css

    def test_css_uses_system_fonts(self, generator, sample_posts):
        """No external font imports."""
        generator.generate_site(sample_posts)
        css = (generator._output_dir / "style.css").read_text(encoding="utf-8")
        assert "googleapis" not in css
        assert "typekit" not in css
        assert "font-face" not in css.lower()


# ---------------------------------------------------------------------------
# HTML structure / semantics
# ---------------------------------------------------------------------------

class TestHTMLSemantics:
    """Test HTML5 semantic structure."""

    def test_all_html_files_have_doctype(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert content.startswith("<!DOCTYPE html>"), f"{html_file.name} missing DOCTYPE"

    def test_all_html_files_have_html_lang(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert 'lang="en"' in content, f"{html_file.name} missing lang attribute"

    def test_all_pages_have_header_main_footer(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert "<header" in content, f"{html_file.name} missing header"
            assert "<main" in content, f"{html_file.name} missing main"
            assert "<footer" in content, f"{html_file.name} missing footer"

    def test_no_tracking_scripts(self, generator, sample_posts):
        """No analytics or tracking scripts."""
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert "google-analytics" not in content.lower()
            assert "gtag" not in content.lower()

    def test_no_external_fonts(self, generator, sample_posts):
        """No external font links in HTML."""
        generator.generate_site(sample_posts)
        for html_file in generator._output_dir.glob("*.html"):
            content = html_file.read_text(encoding="utf-8")
            assert "fonts.googleapis" not in content
            assert "fonts.gstatic" not in content


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_post_with_empty_screenshot(self):
        """Missing screenshot should use placeholder."""
        post = _make_post(screenshot_path="")
        result = _render_post_card(post)
        assert 'src=""' in result or "src=" in result

    def test_post_with_unknown_source(self):
        """Unknown source should get a reasonable label."""
        post = _make_post(source="unknown")
        result = _render_post_card(post)
        assert "via" in result

    def test_post_with_empty_category(self):
        """Empty category should use 'Uncategorized'."""
        post = _make_post(category="")
        result = _render_post_card(post)
        assert "Uncategorized" in result

    def test_many_posts_performance(self):
        """Generating 100 posts should not crash."""
        posts = [_make_post(
            title=f"Site {i}",
            url=f"https://site{i}.com",
            domain=f"site{i}.com",
            category=VALID_CATEGORIES[i % len(VALID_CATEGORIES)],
            discovered_at=f"2025-06-{(i % 28) + 1:02d}T10:00:00+00:00",
        ) for i in range(100)]
        result = _generate_index_page(posts)
        assert "The Daily Web" in result
        assert result.count('class="post-card"') == 11

    def test_post_with_none_values(self):
        """Post with None values should not crash."""
        post = {
            "id": "test",
            "url": "https://example.com",
            "domain": "example.com",
            "title": None,
            "description": None,
            "ai_description": None,
            "category": None,
            "source": None,
            "score": 0,
            "screenshot_path": None,
            "why_interesting": None,
            "discovered_at": "2025-01-01T00:00:00+00:00",
        }
        result = _render_post_card(post)
        assert "<article" in result
        assert "Untitled" in result

    def test_single_category_only(self, generator):
        """Generating with posts in only one category."""
        posts = [_make_post(category="Art")]
        generator.generate_site(posts)
        art_html = (generator._output_dir / "art.html").read_text(encoding="utf-8")
        assert "Art" in art_html
        games_html = (generator._output_dir / "games.html").read_text(encoding="utf-8")
        assert "No Games posts yet" in games_html


# ---------------------------------------------------------------------------
# File content integrity
# ---------------------------------------------------------------------------

class TestFileIntegrity:
    """Test that written files have correct content."""

    def test_index_html_content_matches(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        content = (generator._output_dir / "index.html").read_text(encoding="utf-8")
        expected = generator.generate_index(sample_posts)
        assert content == expected

    def test_archive_html_content_matches(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        content = (generator._output_dir / "archive.html").read_text(encoding="utf-8")
        expected = generator.generate_archive(sample_posts)
        assert content == expected

    def test_category_page_content_matches(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        cat_posts = [p for p in sample_posts if p["category"] == "Interactive"]
        content = (generator._output_dir / "interactive.html").read_text(encoding="utf-8")
        expected = generator.generate_category("Interactive", cat_posts, sample_posts)
        assert content == expected

    def test_css_file_written(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        css = (generator._output_dir / "style.css").read_text(encoding="utf-8")
        assert len(css) > 100
        assert "var(--" in css

    def test_js_file_written(self, generator, sample_posts):
        generator.generate_site(sample_posts)
        js = (generator._output_dir / "app.js").read_text(encoding="utf-8")
        assert "addEventListener" in js
        assert "DOMContentLoaded" in js


# =========================================================================
# Category Filter Tests (dist/js/filter.js) — Playwright browser tests
# =========================================================================

import textwrap as _tw
from playwright.sync_api import Page, expect as _expect

_FILTER_JS = Path(__file__).resolve().parent.parent / "dist" / "js" / "filter.js"

_CATEGORIES_POSTS: dict[str, list[tuple[str, str]]] = {
    "interactive": [
        ("Interactive Post 1", "https://example.com/interactive-1"),
        ("Interactive Post 2", "https://example.com/interactive-2"),
    ],
    "weird": [
        ("Weird Post 1", "https://example.com/weird-1"),
    ],
    "tools": [
        ("Tool Post 1", "https://example.com/tool-1"),
        ("Tool Post 2", "https://example.com/tool-2"),
        ("Tool Post 3", "https://example.com/tool-3"),
    ],
    "games": [
        ("Game Post 1", "https://example.com/game-1"),
    ],
    "art": [
        ("Art Post 1", "https://example.com/art-1"),
        ("Art Post 2", "https://example.com/art-2"),
    ],
    "educational": [
        ("Edu Post 1", "https://example.com/edu-1"),
    ],
}


def _build_filter_html(
    categories: dict[str, list[tuple[str, str]]] | None = None,
) -> str:
    cats = categories if categories is not None else _CATEGORIES_POSTS

    buttons_html = "\n".join(
        f'      <button data-filter="{cat}">{cat.title()}</button>'
        for cat in ["all", *cats.keys()]
    )

    cards_html = "\n".join(
        (
            f'      <article class="post-card" data-category="{cat}">'
            f"<h3>{title}</h3><a href={url}>Visit</a></article>"
        )
        for cat, posts in cats.items()
        for title, url in posts
    )

    return _tw.dedent(f"""\
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Filter Test</title></head>
    <body>
      <div class="filter-bar" role="toolbar" aria-label="Category filters">
{buttons_html}
      </div>
      <div class="posts-grid" data-posts-container>
{cards_html}
      </div>
      <script>{_FILTER_JS.read_text(encoding="utf-8")}</script>
    </body>
    </html>
    """)


_VISIBLE_JS = """() => {
    const cards = document.querySelectorAll('.post-card');
    let visible = 0;
    cards.forEach(c => {
        if (c.style.display !== 'none' && c.style.opacity !== '0') visible++;
    });
    return visible;
}"""


def _visible_count(page: Page) -> int:
    return page.evaluate(_VISIBLE_JS)


_TOTAL_POSTS = 10  # 2 interactive + 1 weird + 3 tools + 1 game + 2 art + 1 edu

_BASE_URL = "data:text/html,<html><head></head><body></body></html>"


@pytest.fixture()
def filter_page(page: Page) -> Page:
    page.goto(_BASE_URL)
    page.set_content(_build_filter_html())
    page.wait_for_function("window.CategoryFilter !== undefined")
    return page


class TestFilterInitialState:
    def test_all_cards_visible_by_default(self, filter_page: Page) -> None:
        assert _visible_count(filter_page) == _TOTAL_POSTS

    def test_all_button_is_active_by_default(self, filter_page: Page) -> None:
        _expect(filter_page.locator('[data-filter="all"]')).to_have_class("active")

    def test_no_empty_state_on_load(self, filter_page: Page) -> None:
        _expect(filter_page.locator("#empty-state")).to_be_hidden()


class TestFilterByCategory:
    def test_click_interactive(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 2
        visible_cats = filter_page.evaluate("""() => {
            const cards = document.querySelectorAll('.post-card');
            const cats = [];
            cards.forEach(c => {
                if (c.style.display !== 'none') cats.push(c.dataset.category);
            });
            return cats;
        }""")
        assert all(c == "interactive" for c in visible_cats)

    def test_click_weird(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="weird"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 1

    def test_click_tools_shows_three(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 3

    def test_click_games(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="games"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 1

    def test_click_art_shows_two(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="art"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 2

    def test_click_educational(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="educational"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 1


class TestFilterAllButton:
    def test_all_after_interactive(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == 2

        filter_page.locator('[data-filter="all"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == _TOTAL_POSTS

    def test_all_after_tools(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        filter_page.wait_for_timeout(400)

        filter_page.locator('[data-filter="all"]').click()
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == _TOTAL_POSTS


class TestFilterActiveButton:
    def test_active_class_moves(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        _expect(filter_page.locator('[data-filter="interactive"]')).to_have_class("active")
        _expect(filter_page.locator('[data-filter="all"]')).not_to_have_class("active")

    def test_active_returns_to_all(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="art"]').click()
        filter_page.locator('[data-filter="all"]').click()
        _expect(filter_page.locator('[data-filter="all"]')).to_have_class("active")
        _expect(filter_page.locator('[data-filter="art"]')).not_to_have_class("active")


class TestFilterUrlHash:
    def test_hash_updates_on_click(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        hash_val = filter_page.evaluate("() => window.location.hash")
        assert hash_val == "#interactive"

    def test_hash_cleared_when_all(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        assert filter_page.evaluate("() => window.location.hash") == "#tools"

        filter_page.locator('[data-filter="all"]').click()
        hash_val = filter_page.evaluate("() => window.location.hash")
        assert hash_val == "" or hash_val is None or hash_val == "#"

    def test_filter_from_initial_hash(self, filter_page: Page) -> None:
        filter_page.set_content(_build_filter_html())
        filter_page.evaluate("() => { history.replaceState(null, '', '#tools'); }")
        filter_page.evaluate("() => window.CategoryFilter.init()")
        filter_page.wait_for_timeout(400)

        assert _visible_count(filter_page) == 3
        _expect(filter_page.locator('[data-filter="tools"]')).to_have_class("active")


class TestFilterEmptyState:
    def test_empty_state_shows_when_no_cards_match(self, filter_page: Page) -> None:
        filter_page.evaluate("""() => {
            document.querySelectorAll('.post-card[data-category="weird"]').forEach(c => c.remove());
            window.CategoryFilter.init();
        }""")
        filter_page.locator('[data-filter="weird"]').click()
        filter_page.wait_for_timeout(400)

        _expect(filter_page.locator("#empty-state")).to_be_visible()
        _expect(filter_page.locator("#empty-state")).to_have_text("No posts in this category")

    def test_empty_state_hidden_when_cards_exist(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        filter_page.wait_for_timeout(400)
        _expect(filter_page.locator("#empty-state")).to_be_hidden()

    def test_empty_state_hidden_after_showing_all(self, filter_page: Page) -> None:
        filter_page.evaluate("""() => {
            document.querySelectorAll('.post-card[data-category="weird"]').forEach(c => c.remove());
            window.CategoryFilter.init();
        }""")
        filter_page.locator('[data-filter="weird"]').click()
        filter_page.wait_for_timeout(400)
        _expect(filter_page.locator("#empty-state")).to_be_visible()

        filter_page.locator('[data-filter="all"]').click()
        filter_page.wait_for_timeout(400)
        _expect(filter_page.locator("#empty-state")).to_be_hidden()


class TestFilterTransitions:
    def test_hidden_card_display_none(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        filter_page.wait_for_timeout(400)

        _expect(filter_page.locator('.post-card[data-category="weird"]').first).to_have_css("display", "none")

    def test_shown_card_full_opacity(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="interactive"]').click()
        filter_page.wait_for_timeout(400)

        opacity = filter_page.locator('.post-card[data-category="interactive"]').first.evaluate(
            "el => getComputedStyle(el).opacity"
        )
        assert float(opacity) > 0.9


class TestFilterRapidClicks:
    def test_rapid_switching(self, filter_page: Page) -> None:
        for cat in ["tools", "art", "games", "weird", "all"]:
            filter_page.locator(f'[data-filter="{cat}"]').click()

        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == _TOTAL_POSTS
        _expect(filter_page.locator('[data-filter="all"]')).to_have_class("active")


class TestFilterPublicAPI:
    def test_api_exists(self, filter_page: Page) -> None:
        assert filter_page.evaluate("typeof window.CategoryFilter.init") == "function"
        assert filter_page.evaluate("typeof window.CategoryFilter.applyFilter") == "function"
        assert filter_page.evaluate("typeof window.CategoryFilter.updateUrlHash") == "function"
        assert filter_page.evaluate("typeof window.CategoryFilter.getCategoryFromHash") == "function"

    def test_valid_categories_list(self, filter_page: Page) -> None:
        cats = filter_page.evaluate("window.CategoryFilter.VALID_CATEGORIES")
        expected = {"all", "interactive", "weird", "tools", "games", "art", "educational"}
        assert set(cats) == expected

    def test_empty_message_constant(self, filter_page: Page) -> None:
        assert filter_page.evaluate("window.CategoryFilter.EMPTY_MESSAGE") == "No posts in this category"


class TestFilterEdgeCases:
    def test_empty_page_no_crash(self, filter_page: Page) -> None:
        html = '<html><body><div class="posts-grid" data-posts-container></div></body></html>'
        filter_page.goto(_BASE_URL)
        filter_page.set_content(html)
        filter_page.evaluate(_FILTER_JS.read_text(encoding="utf-8"))

    def test_no_post_cards(self, filter_page: Page) -> None:
        html = _tw.dedent("""\
        <html><body>
          <div class="filter-bar">
            <button data-filter="all">All</button>
            <button data-filter="tools">Tools</button>
          </div>
          <div class="posts-grid" data-posts-container></div>
          <script>""" + _FILTER_JS.read_text(encoding="utf-8") + """</script>
        </body></html>""")
        filter_page.goto(_BASE_URL)
        filter_page.set_content(html)

    def test_invalid_hash_ignored(self, filter_page: Page) -> None:
        filter_page.goto(_BASE_URL)
        filter_page.set_content(_build_filter_html())
        filter_page.evaluate('() => { window.location.hash = "foobar"; }')
        filter_page.evaluate("() => window.CategoryFilter.init()")
        filter_page.wait_for_timeout(400)
        assert _visible_count(filter_page) == _TOTAL_POSTS

    def test_switch_cleans_previous(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        filter_page.wait_for_timeout(400)

        filter_page.locator('[data-filter="art"]').click()
        filter_page.wait_for_timeout(400)

        tools_visible = filter_page.evaluate("""() => {
            const cards = document.querySelectorAll('.post-card[data-category="tools"]');
            let v = 0;
            cards.forEach(c => { if (c.style.display !== 'none') v++; });
            return v;
        }""")
        assert tools_visible == 0


class TestFilterAccessibility:
    def test_all_aria_pressed_true_default(self, filter_page: Page) -> None:
        _expect(filter_page.locator('[data-filter="all"]')).to_have_attribute("aria-pressed", "true")

    def test_other_aria_pressed_false_default(self, filter_page: Page) -> None:
        _expect(filter_page.locator('[data-filter="tools"]')).to_have_attribute("aria-pressed", "false")

    def test_aria_pressed_toggles(self, filter_page: Page) -> None:
        filter_page.locator('[data-filter="tools"]').click()
        _expect(filter_page.locator('[data-filter="tools"]')).to_have_attribute("aria-pressed", "true")
        _expect(filter_page.locator('[data-filter="all"]')).to_have_attribute("aria-pressed", "false")
