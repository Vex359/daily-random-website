"""Static site generator.

Generates the final HTML/CSS/JS website from scored and filtered
content, ready for deployment to Netlify.

Usage:
    python -m src.site.generator

Produces:
    dist/index.html        – homepage with hero + recent grid
    dist/archive.html      – all posts with filter buttons
    dist/{category}.html   – per-category filtered pages
    dist/style.css         – responsive styles
    dist/app.js            – client-side category filter
"""

from __future__ import annotations

import html
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.ai.generator import clean_website_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITE_NAME = "The Daily Web"
SITE_TAGLINE = "Discover something weird and wonderful on the internet every day."
SITE_DESCRIPTION = (
    "A curated collection of the internet's most interesting, weird, "
    "and interactive websites. Updated daily."
)
RECENT_COUNT = 10
SCREENSHOT_PLACEHOLDER = "screenshots/placeholder.webp"

VALID_CATEGORIES: tuple[str, ...] = (
    "Interactive",
    "Weird",
    "Tools",
    "Games",
    "Art",
    "Educational",
)

CATEGORY_SLUGS: dict[str, str] = {
    "Interactive": "interactive",
    "Weird": "weird",
    "Tools": "tools",
    "Games": "games",
    "Art": "art",
    "Educational": "educational",
}

POPULAR_TAGS: tuple[str, ...] = (
    "3D",
    "AI",
    "Analytics",
    "Audio",
    "Database",
    "Design",
    "Developer Tools",
    "Open Source",
    "Productivity",
    "Simulation",
    "Voice AI",
    "Visualization",
)


# ---------------------------------------------------------------------------
# HTML helpers (escape, no frameworks)
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text), quote=True)


def _esc_display(text: str) -> str:
    """HTML-escape for visible text content."""
    return html.escape(str(text))


# ---------------------------------------------------------------------------
# Post card renderer
# ---------------------------------------------------------------------------

def _render_post_card(post: dict[str, Any]) -> str:
    """Render a single post as an Instagram-style card."""
    raw_title = post.get("clean_title") or post.get("title")
    domain = str(post.get("domain") or "")
    if raw_title is None or raw_title == "":
        title = "Untitled"
    else:
        title = _esc_display(clean_website_name(raw_title, domain))
    description = _esc_display(post.get("ai_description") or post.get("description") or "")
    category = _esc_display(post.get("category") or "Uncategorized")
    source = _esc_display(post.get("source") or "")
    url = _esc(post.get("url") or "#")
    screenshot = _esc(post.get("screenshot_path") or SCREENSHOT_PLACEHOLDER)
    cat_str = str(post.get("category") or "")
    category_lower = CATEGORY_SLUGS.get(cat_str, cat_str.lower().replace(" ", "-"))
    alt_text = _esc(f"Screenshot of {domain or title}")

    source_label = _source_label(source)

    tags = post.get("tags") or []
    if isinstance(tags, list):
        tags_lower_list = [str(t).lower().replace(" ", "-") for t in tags if t]
    else:
        tags_lower_list = []
    data_tags = _esc(",".join(tags_lower_list))

    tags_html = ""
    if tags and isinstance(tags, list):
        tag_spans = [
            f'<span class="post-tag-pill" data-tag-slug="{_esc(str(t).lower().replace(" ", "-"))}">#{_esc_display(str(t))}</span>'
            for t in tags[:4] if t
        ]
        if tag_spans:
            tags_html = f'<div class="post-tags-list">{" ".join(tag_spans)}</div>'

    return f"""<article class="post-card" data-category="{category_lower}" data-tags="{data_tags}">
      <img src="{screenshot}" alt="{alt_text}" loading="lazy">
      <div class="card-content">
        <h3 class="post-title"><strong>{title}</strong></h3>
        <p class="post-description">{description}</p>
        {tags_html}
        <div class="post-meta">
          <span class="post-category">{category}</span>
          <span class="post-source">via {source_label}</span>
        </div>
        <a href="{url}" class="visit-btn" target="_blank" rel="noopener">Visit Website</a>
      </div>
    </article>"""


def _source_label(source: str) -> str:
    """Map source key to human-readable label."""
    labels = {
        "hackernews": "Hacker News",
        "github": "GitHub",
        "rss": "RSS Feed",
        "wikipedia": "Wikipedia",
        "awesome_lists": "Awesome Lists",
    }
    if not source:
        return "Unknown"
    if source in labels:
        return labels[source]
    return source.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Filter buttons partial
# ---------------------------------------------------------------------------

def _render_filter_buttons(active: str = "all") -> str:
    """Render the category filter button bar."""
    buttons = [("All", "all")]
    for cat in VALID_CATEGORIES:
        buttons.append((cat, CATEGORY_SLUGS[cat]))

    items = []
    for label, slug in buttons:
        cls = "active" if slug == active else ""
        items.append(
            f'      <button class="filter-btn {cls}" data-filter="{slug}">{_esc_display(label)}</button>'
        )
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Page shells
# ---------------------------------------------------------------------------

def _html_head(title: str, css_path: str = "style.css") -> str:
    """Return the <head> block."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc_display(title)}</title>
  <link rel="stylesheet" href="{css_path}">
</head>"""


def _site_header(active_page: str = "", extra: str = "") -> str:
    """Render the site header with nav."""
    nav_links = [
        ("Home", "index.html", "home"),
        ("Archive", "archive.html", "archive"),
    ]
    nav_items = []
    for label, href, slug in nav_links:
        cls = ' class="active"' if slug == active_page else ""
        nav_items.append(f'      <a href="{href}"{cls}>{_esc_display(label)}</a>')
    nav_html = "\n".join(nav_items)

    extra_html = f"\n    {_esc_display(extra)}" if extra else ""

    return f"""  <header class="site-header">
    <div class="header-content">
      <a href="index.html" class="site-name">{_esc_display(SITE_NAME)}</a>{extra_html}
      <nav class="main-nav">
{nav_html}
      </nav>
    </div>
  </header>"""


def _site_footer() -> str:
    """Render the site footer."""
    year = datetime.now(timezone.utc).year
    return f"""  <footer class="site-footer">
    <div class="footer-content">
      <p>{_esc_display(SITE_DESCRIPTION)}</p>
      <p class="footer-copy">&copy; {year} {_esc_display(SITE_NAME)} &middot; Curated daily with care</p>
    </div>
  </footer>
  <script src="app.js"></script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------

def _generate_index_page(posts: list[dict[str, Any]]) -> str:
    """Generate the homepage HTML."""
    # Sort by discovered_at descending
    sorted_posts = sorted(
        posts, key=lambda p: p.get("discovered_at", ""), reverse=True
    )
    latest = sorted_posts[0] if sorted_posts else None
    recent = sorted_posts[:RECENT_COUNT]

    # Hero card (today's discovery)
    hero_html = ""
    if latest:
        hero_html = f"""  <main class="site-main">
    <section class="hero-section">
      <h2>Today&rsquo;s Discovery</h2>
      {_render_post_card(latest)}
    </section>"""
    else:
        hero_html = f"""  <main class="site-main">
    <section class="hero-section">
      <h2>Today&rsquo;s Discovery</h2>
      <div class="empty-state">
        <p>No discoveries yet. Check back soon!</p>
      </div>
    </section>"""

    # Recent discoveries grid
    grid_cards = []
    for post in recent:
        grid_cards.append(_render_post_card(post))
    grid_html = "\n".join(grid_cards)

    if recent:
        recent_section = f"""    <section class="recent-section">
      <h2>Recent Discoveries</h2>
      <div class="posts-grid">
{grid_html}
      </div>
    </section>"""
    else:
        recent_section = """    <section class="recent-section">
      <h2>Recent Discoveries</h2>
      <div class="empty-state">
        <p>No recent discoveries. The pipeline will populate this soon.</p>
      </div>
    </section>"""

    return f"""{_html_head(SITE_NAME)}
<body>
{_site_header("home")}
{hero_html}
{recent_section}
{_site_footer()}"""


def _render_filter_dropdown(posts: list[dict[str, Any]], active: str = "all") -> str:
    """Render a clean <select> dropdown containing all categories and tags."""
    lines = ['      <div class="filter-dropdown-wrap">']
    lines.append('        <label for="category-select" class="filter-label">🏷️ Filter by Tag / Category:</label>')
    lines.append('        <select id="category-select" class="filter-select">')
    lines.append('          <option value="all">🌟 All Discoveries</option>')
    lines.append('          <optgroup label="Categories">')
    seen_slugs = set()
    for cat in VALID_CATEGORIES:
        slug = CATEGORY_SLUGS[cat]
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            sel = ' selected="selected"' if slug == active else ""
            lines.append(f'            <option value="{slug}"{sel}>{_esc_display(cat)}</option>')
    lines.append('          </optgroup>')

    # Unique tags from posts + popular topics
    tag_options = set(POPULAR_TAGS)
    for p in posts:
        tags = p.get("tags")
        if isinstance(tags, list):
            for t in tags:
                if t and t not in VALID_CATEGORIES:
                    tag_options.add(str(t))

    if tag_options:
        lines.append('          <optgroup label="Topic Tags">')
        for t in sorted(list(tag_options)):
            slug = str(t).lower().replace(" ", "-")
            sel = ' selected="selected"' if slug == active else ""
            lines.append(f'            <option value="{slug}"{sel}>#{_esc_display(str(t))}</option>')
        lines.append('          </optgroup>')

    lines.append('        </select>')
    lines.append('      </div>')
    return "\n".join(lines)


def _generate_archive_page(posts: list[dict[str, Any]]) -> str:
    """Generate the archive page HTML."""
    sorted_posts = sorted(
        posts, key=lambda p: p.get("discovered_at", ""), reverse=True
    )

    dropdown_html = _render_filter_dropdown(sorted_posts, "all")
    buttons_html = _render_filter_buttons("all")

    if sorted_posts:
        cards_html = "\n".join(_render_post_card(p) for p in sorted_posts)
        posts_section = f"""    <section class="archive-section">
      <div class="posts-grid">
{cards_html}
      </div>
      <div id="filter-empty-state" class="empty-state" style="display:none;">
        <p>No discoveries match this tag yet. Check back soon!</p>
      </div>
    </section>"""
    else:
        posts_section = """    <section class="archive-section">
      <div class="empty-state">
        <p>No posts yet. The pipeline will populate this soon.</p>
      </div>
    </section>"""

    return f"""{_html_head(f"Archive | {SITE_NAME}")}
<body>
{_site_header("archive")}
  <main class="site-main">
    <section class="filter-bar">
      <div class="filter-header">
        <h2>All Discoveries</h2>
{dropdown_html}
      </div>
      <div class="filter-buttons">
{buttons_html}
      </div>
    </section>
{posts_section}
{_site_footer()}"""


def _generate_category_page(
    category: str, posts: list[dict[str, Any]], all_posts: list[dict[str, Any]]
) -> str:
    """Generate a category-filtered page HTML."""
    slug = CATEGORY_SLUGS.get(category, category.lower())
    filtered = sorted(
        posts, key=lambda p: p.get("discovered_at", ""), reverse=True
    )

    if filtered:
        cards_html = "\n".join(_render_post_card(p) for p in filtered)
        posts_section = f"""    <section class="category-section">
      <div class="posts-grid">
{cards_html}
      </div>
    </section>"""
    else:
        posts_section = f"""    <section class="category-section">
      <div class="empty-state">
        <p>No {_esc_display(category)} posts yet. Check back soon!</p>
      </div>
    </section>"""

    return f"""{_html_head(f"{category} | {SITE_NAME}")}
<body>
{_site_header("archive", category)}
  <main class="site-main">
    <section class="filter-bar">
      <div class="filter-buttons">
{_render_filter_buttons(slug)}
      </div>
    </section>
{posts_section}
{_site_footer()}"""


def _generate_post_page(post: dict[str, Any]) -> str:
    """Generate an individual post detail page."""
    title = _esc_display(post.get("title") or "Untitled")
    description = _esc_display(post.get("ai_description") or post.get("description") or "")
    category = _esc_display(post.get("category") or "Uncategorized")
    source = _esc_display(_source_label(post.get("source") or ""))
    url = _esc(post.get("url") or "#")
    domain = _esc_display(post.get("domain") or "")
    screenshot = _esc(post.get("screenshot_path") or SCREENSHOT_PLACEHOLDER)
    why = _esc_display(post.get("why_interesting") or "")

    return f"""{_html_head(f"{title} | {SITE_NAME}")}
<body>
{_site_header()}
  <main class="site-main">
    <article class="post-detail">
      <div class="post-detail-header">
        <span class="post-category">{category}</span>
        <span class="post-source">via {source}</span>
      </div>
      <h1 class="post-detail-title">{title}</h1>
      <div class="post-detail-image">
        <img src="{screenshot}" alt="{_esc(f'Screenshot of {domain or title}')}" loading="lazy">
      </div>
      <p class="post-detail-description">{description}</p>
      {f'<p class="post-detail-why"><strong>Why interesting:</strong> {why}</p>' if why else ''}
      <a href="{url}" class="visit-btn visit-btn--large" target="_blank" rel="noopener">Visit Website</a>
    </article>
  </main>
{_site_footer()}"""


# ---------------------------------------------------------------------------
# Static assets (CSS + JS)
# ---------------------------------------------------------------------------

_CSS = """\
/* ======================================================
   The Daily Web — Styles
   System fonts only, responsive, no external deps
   ====================================================== */

/* --- Reset --- */
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* --- Variables --- */
:root {
  --color-bg: #fafafa;
  --color-surface: #ffffff;
  --color-text: #1a1a2e;
  --color-text-muted: #6b7280;
  --color-primary: #2563eb;
  --color-primary-hover: #1d4ed8;
  --color-border: #e5e7eb;
  --color-accent: #f59e0b;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
  --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
  --radius: 12px;
  --max-width: 1200px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

body {
  font-family: var(--font-sans);
  background: var(--color-bg);
  color: var(--color-text);
  line-height: 1.6;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

a { color: var(--color-primary); text-decoration: none; }
a:hover { text-decoration: underline; }

img { display: block; max-width: 100%; height: auto; }

/* --- Header --- */
.site-header {
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: var(--shadow-sm);
}

.header-content {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 1rem 1.5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.site-name {
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--color-text);
  letter-spacing: -0.02em;
}
.site-name:hover { text-decoration: none; color: var(--color-primary); }

.main-nav { display: flex; gap: 1rem; }
.main-nav a {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--color-text-muted);
  padding: 0.25rem 0;
  border-bottom: 2px solid transparent;
  transition: color 0.2s, border-color 0.2s;
}
.main-nav a:hover { color: var(--color-text); text-decoration: none; }
.main-nav a.active {
  color: var(--color-primary);
  border-bottom-color: var(--color-primary);
}

/* --- Main --- */
.site-main {
  flex: 1;
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 2rem 1.5rem;
  width: 100%;
}

/* --- Hero --- */
.hero-section h2,
.recent-section h2,
.archive-section h2,
.filter-bar h2 {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 1.25rem;
  letter-spacing: -0.01em;
}

.hero-section { margin-bottom: 3rem; }

/* --- Post card (Instagram-style) --- */
.post-card {
  background: var(--color-surface);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow-md);
  transition: transform 0.2s, box-shadow 0.2s;
  display: flex;
  flex-direction: column;
}
.post-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.post-card img {
  width: 100%;
  aspect-ratio: 16/9;
  object-fit: cover;
  background: var(--color-border);
}

.card-content { padding: 1rem 1.25rem 1.25rem; display: flex; flex-direction: column; gap: 0.5rem; flex: 1; }
.post-title { font-size: 1.1rem; font-weight: 700; line-height: 1.3; }
.post-description {
  font-size: 0.875rem;
  color: var(--color-text-muted);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.post-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.75rem;
  color: var(--color-text-muted);
  margin-top: auto;
  padding-top: 0.5rem;
}
.post-category {
  background: var(--color-primary);
  color: #fff;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.visit-btn {
  display: inline-block;
  margin-top: 0.75rem;
  padding: 0.5rem 1rem;
  background: var(--color-primary);
  color: #fff;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  text-align: center;
  transition: background 0.2s;
}
.visit-btn:hover { background: var(--color-primary-hover); text-decoration: none; }
.visit-btn--large { padding: 0.75rem 2rem; font-size: 1rem; }

/* --- Posts grid --- */
.posts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
}

/* --- Filter bar --- */
.filter-bar {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  margin-bottom: 2rem;
}

.filter-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
  width: 100%;
}

.filter-dropdown-wrap {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}

.filter-label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--color-text-muted);
}

.filter-select {
  padding: 0.45rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  outline: none;
  font-family: var(--font-sans);
  box-shadow: var(--shadow-sm);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.filter-select:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}

.filter-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.filter-btn {
  padding: 0.4rem 0.85rem;
  border: 1px solid var(--color-border);
  border-radius: 999px;
  background: var(--color-surface);
  color: var(--color-text-muted);
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s, color 0.2s, border-color 0.2s;
  font-family: var(--font-sans);
}
.filter-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}
.filter-btn.active {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}

.post-title {
  font-size: 1.15rem;
  line-height: 1.35;
}

.post-title strong {
  font-weight: 800;
  color: var(--color-text);
}

.post-tags-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 0.2rem 0;
}

.post-tag-pill {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  color: #4338ca;
  background: #eef2ff;
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.post-tag-pill:hover {
  background: #e0e7ff;
  color: #3730a3;
}

/* --- Post detail --- */
.post-detail {
  max-width: 720px;
  margin: 0 auto;
}
.post-detail-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.post-detail-title {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: 1.5rem;
  letter-spacing: -0.02em;
}
.post-detail-image {
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-md);
}
.post-detail-image img { width: 100%; }
.post-detail-description {
  font-size: 1.1rem;
  line-height: 1.7;
  margin-bottom: 1.5rem;
}
.post-detail-why {
  font-size: 0.95rem;
  color: var(--color-text-muted);
  margin-bottom: 2rem;
  padding: 1rem;
  background: var(--color-bg);
  border-radius: 8px;
  border-left: 3px solid var(--color-accent);
}

/* --- Empty state --- */
.empty-state {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--color-text-muted);
}
.empty-state p { font-size: 1rem; }

/* --- Footer --- */
.site-footer {
  background: var(--color-surface);
  border-top: 1px solid var(--color-border);
  margin-top: auto;
}
.footer-content {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 2rem 1.5rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--color-text-muted);
}
.footer-content p { margin-bottom: 0.5rem; }
.footer-copy { font-size: 0.75rem; opacity: 0.7; }

/* === Responsive === */
@media (max-width: 768px) {
  .header-content { flex-direction: column; text-align: center; }
  .site-main { padding: 1.25rem 1rem; }
  .posts-grid { grid-template-columns: 1fr; }
  .hero-section .post-card img { aspect-ratio: 16/9; }
  .filter-bar { flex-direction: column; align-items: flex-start; }
  .filter-header { flex-direction: column; align-items: flex-start; }
  .post-detail-title { font-size: 1.5rem; }
}
@media (min-width: 769px) and (max-width: 1024px) {
  .posts-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

_JS = """\
// Category & Tag filter for archive and category pages
document.addEventListener('DOMContentLoaded', function() {
  var buttons = document.querySelectorAll('.filter-btn');
  var dropdown = document.getElementById('category-select');
  var cards = document.querySelectorAll('.post-card');

  if (!cards.length) return;

  function applyFilter(filter) {
    if (!filter) filter = 'all';
    filter = filter.toLowerCase();

    // Update active button state
    buttons.forEach(function(b) {
      if (b.getAttribute('data-filter') === filter) {
        b.classList.add('active');
      } else {
        b.classList.remove('active');
      }
    });

    // Update dropdown selection
    if (dropdown) {
      for (var i = 0; i < dropdown.options.length; i++) {
        if (dropdown.options[i].value.toLowerCase() === filter) {
          dropdown.selectedIndex = i;
          break;
        }
      }
    }

    // Filter cards
    var visible = 0;
    cards.forEach(function(card) {
      var cat = (card.getAttribute('data-category') || '').toLowerCase();
      var rawTags = (card.getAttribute('data-tags') || '').toLowerCase();
      var tagsList = rawTags ? rawTags.split(',') : [];
      var matches = (filter === 'all') || (cat === filter) || (tagsList.indexOf(filter) !== -1);
      if (matches) {
        card.style.display = '';
        visible++;
      } else {
        card.style.display = 'none';
      }
    });

    var empty = document.getElementById('filter-empty-state');
    if (empty) {
      empty.style.display = visible === 0 ? 'block' : 'none';
    }
  }

  buttons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      applyFilter(this.getAttribute('data-filter'));
    });
  });

  if (dropdown) {
    dropdown.addEventListener('change', function() {
      applyFilter(this.value);
    });
  }

  var tagPills = document.querySelectorAll('.post-tag-pill');
  tagPills.forEach(function(pill) {
    pill.addEventListener('click', function(e) {
      e.preventDefault();
      var tag = this.getAttribute('data-tag-slug');
      if (tag) applyFilter(tag);
    });
  });
});
"""


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class SiteGenerator:
    """Generates a static HTML site from post data.

    Produces index.html, archive.html, category pages, and optional
    individual post pages.  Also writes style.css and app.js to the
    output directory.

    Usage::

        gen = SiteGenerator()
        gen.generate_site(posts, output_dir="dist")
    """

    def __init__(self, *, output_dir: str | Path = "dist") -> None:
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_site(
        self, posts: list[dict[str, Any]], output_dir: str | Path | None = None
    ) -> Path:
        """Generate all site files.

        Args:
            posts: List of post dicts. Each must have at minimum:
                url, title, category, discovered_at.
            output_dir: Override output directory (default: self._output_dir).

        Returns:
            Path to the output directory.
        """
        out = Path(output_dir) if output_dir else self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Generating site with %d posts to %s", len(posts), out)

        # Static assets
        self._write_file(out / "style.css", _CSS)
        self._write_file(out / "app.js", _JS)

        # Pages
        index_html = self.generate_index(posts)
        self._write_file(out / "index.html", index_html)

        archive_html = self.generate_archive(posts)
        self._write_file(out / "archive.html", archive_html)

        # Category pages
        for category in VALID_CATEGORIES:
            cat_posts = [p for p in posts if p.get("category") == category]
            cat_html = self.generate_category(category, cat_posts, all_posts=posts)
            slug = CATEGORY_SLUGS[category]
            self._write_file(out / f"{slug}.html", cat_html)

        # Optional: individual post pages
        for post in posts:
            if post.get("url"):
                post_html = self.generate_post_page(post)
                slug = self._post_slug(post)
                self._write_file(out / f"{slug}.html", post_html)

        logger.info("Site generation complete: %s", out)
        return out

    def generate_index(self, posts: list[dict[str, Any]]) -> str:
        """Generate homepage HTML string."""
        return _generate_index_page(posts)

    def generate_archive(self, posts: list[dict[str, Any]]) -> str:
        """Generate archive page HTML string."""
        return _generate_archive_page(posts)

    def generate_category(
        self,
        category: str,
        posts: list[dict[str, Any]],
        all_posts: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a category page HTML string.

        Args:
            category: Category name (e.g. "Interactive").
            posts: Posts filtered to this category.
            all_posts: All posts (for future use, e.g. counts).
        """
        return _generate_category_page(category, posts, all_posts or posts)

    def generate_post_page(self, post: dict[str, Any]) -> str:
        """Generate an individual post detail page HTML string."""
        return _generate_post_page(post)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_file(self, path: Path, content: str) -> None:
        """Write content to a file, logging the action."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote %s", path)

    def _post_slug(self, post: dict[str, Any]) -> str:
        """Generate a filesystem-safe slug from a post."""
        domain = post.get("domain", "")
        if domain:
            # e.g. "cool-site.com" -> "post_cool-site-com"
            return f"post_{domain.replace('.', '-').replace('/', '-')}"
        url = post.get("url", "unknown")
        # Fallback: use a hash-like approach
        import hashlib
        digest = hashlib.md5(url.encode()).hexdigest()[:12]
        return f"post_{digest}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _demo_posts() -> list[dict[str, Any]]:
    """Create sample posts for the demo."""
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "id": "demo-1",
            "url": "https://neal.fun",
            "domain": "neal.fun",
            "title": "Neal.fun - Fun Web Experiments",
            "description": "A collection of fun interactive web experiments.",
            "ai_description": "Dive into a collection of delightfully addictive browser experiments. From the Password Game to Spend Bill Gates' Money, each one is polished and surprisingly fun. Warning: you might lose hours.",
            "category": "Interactive",
            "source": "hackernews",
            "score": 88,
            "screenshot_path": "screenshots/neal-fun.webp",
            "why_interesting": "Creative interactive experiments with excellent UX",
            "discovered_at": now,
        },
        {
            "id": "demo-2",
            "url": "https://window-swap.com",
            "domain": "window-swap.com",
            "title": "Window Swap",
            "description": "Open a window to someone else's view.",
            "ai_description": "Peek through someone else's window from anywhere in the world. People submit videos of their views and you can swap between them. A surprisingly calming way to travel without moving.",
            "category": "Weird",
            "source": "rss",
            "score": 72,
            "screenshot_path": "screenshots/window-swap.webp",
            "why_interesting": "Beautiful concept connecting strangers through window views",
            "discovered_at": now,
        },
        {
            "id": "demo-3",
            "url": "https://excalidraw.com",
            "domain": "excalidraw.com",
            "title": "Excalidraw - Virtual Whiteboard",
            "description": "A virtual whiteboard for sketching diagrams.",
            "ai_description": "Sketch hand-drawn diagrams that look like they came from a notebook. Real-time collaboration, end-to-end encryption, and exports to PNG, SVG, and JSON. The tool you didn't know you needed.",
            "category": "Tools",
            "source": "hackernews",
            "score": 85,
            "screenshot_path": "screenshots/excalidraw.webp",
            "why_interesting": "Perfect blend of simplicity and functionality",
            "discovered_at": now,
        },
        {
            "id": "demo-4",
            "url": "https://indie-game.fun/roguelike",
            "domain": "indie-game.fun",
            "title": "Roguelike Dungeon Crawler",
            "description": "A browser-based roguelike dungeon crawler.",
            "ai_description": "A browser roguelike that proves you don't need fancy graphics to be addictive. Procedurally generated dungeons, permadeath, and just one more run. Classic gameplay, modern web tech.",
            "category": "Games",
            "source": "hackernews",
            "score": 75,
            "screenshot_path": "screenshots/indie-game.webp",
            "why_interesting": "Innovative browser game with retro roguelike mechanics",
            "discovered_at": now,
        },
        {
            "id": "demo-5",
            "url": "https://creative-art.xyz/generative",
            "domain": "creative-art.xyz",
            "title": "Generative Art Engine",
            "description": "A generative art creation tool.",
            "ai_description": "Watch algorithms paint masterpieces in real time. This generative art engine creates unique visual pieces you can tweak and explore. Each click produces something entirely new.",
            "category": "Art",
            "source": "github",
            "score": 68,
            "screenshot_path": "screenshots/creative-art.webp",
            "why_interesting": "Beautiful generative art with real-time interaction",
            "discovered_at": now,
        },
        {
            "id": "demo-6",
            "url": "https://learn-coding.dev/python",
            "domain": "learn-coding.dev",
            "title": "Interactive Python Tutorial",
            "description": "Learn Python with hands-on interactive exercises.",
            "ai_description": "Learn Python by actually writing code, not just reading about it. Interactive exercises that run in your browser with instant feedback. The fastest way from zero to your first script.",
            "category": "Educational",
            "source": "github",
            "score": 65,
            "screenshot_path": "screenshots/learn-coding.webp",
            "why_interesting": "Excellent interactive learning experience for beginners",
            "discovered_at": now,
        },
        {
            "id": "demo-7",
            "url": "https://weird-science.org",
            "domain": "weird-science.org",
            "title": "Bizarre Science Experiments",
            "description": "Strange experiments you can try at home.",
            "ai_description": "A collection of experiments that make you question reality. From non-Newtonian fluids to magnetic putty, these projects are equal parts educational and weirdly satisfying.",
            "category": "Weird",
            "source": "wikipedia",
            "score": 70,
            "screenshot_path": "screenshots/weird-science.webp",
            "why_interesting": "Fascinating experiments that challenge intuition",
            "discovered_at": now,
        },
        {
            "id": "demo-8",
            "url": "https://tiny-utility.me/json-formatter",
            "domain": "tiny-utility.me",
            "title": "JSON Formatter & Validator",
            "description": "A simple JSON formatting tool.",
            "ai_description": "Paste your messy JSON and get beautifully formatted output instantly. Syntax highlighting, error detection, and minification. Simple, fast, no ads. Exactly what a tool should be.",
            "category": "Tools",
            "source": "hackernews",
            "score": 60,
            "screenshot_path": "screenshots/json-formatter.webp",
            "why_interesting": "Minimalist tool done right",
            "discovered_at": now,
        },
    ]


def main() -> None:
    """CLI entry point: generate demo site."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    posts = _demo_posts()
    out_dir = Path("dist")

    gen = SiteGenerator(output_dir=out_dir)
    result = gen.generate_site(posts)

    # Verify output
    expected = [
        "index.html", "archive.html", "style.css", "app.js",
        "interactive.html", "weird.html", "tools.html",
        "games.html", "art.html", "educational.html",
    ]
    print(f"\n{'='*60}")
    print("STATIC SITE GENERATOR — DEMO")
    print(f"{'='*60}")
    print(f"\nOutput: {result}")
    print(f"Posts:  {len(posts)}")

    all_ok = True
    for filename in expected:
        path = result / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status:7s}] {filename:25s} ({size:,} bytes)")
        if not exists:
            all_ok = False

    # Verify post pages
    post_pages = list(result.glob("post_*.html"))
    print(f"\n  Post pages: {len(post_pages)}")

    # Verify HTML structure
    index_html = (result / "index.html").read_text(encoding="utf-8")
    checks = [
        ("DOCTYPE html", "<!DOCTYPE html>" in index_html),
        ("meta viewport", 'name="viewport"' in index_html),
        ("semantic header", "<header" in index_html),
        ("semantic main", "<main" in index_html),
        ("semantic footer", "<footer" in index_html),
        ("site name", "The Daily Web" in index_html),
        ("links style.css", 'href="style.css"' in index_html),
        ("links app.js", 'src="app.js"' in index_html),
    ]

    print(f"\n{'='*60}")
    print("HTML VALIDATION")
    print(f"{'='*60}")
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
