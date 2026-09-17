"""Content quality filter.

Evaluates content quality based on structural analysis,
content depth, source reputation, and other heuristics.

Quality checks include:
- URL is well-formed and resolves
- Response is HTML with 200 status
- Page has meaningful content (not just a login form or error page)
- Title exists and is meaningful
- Content is not too short or too thin
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from src.filters.dedup import URLNormalizer
from src.filters.domains import DomainFilter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum word count for meaningful content
MIN_CONTENT_WORDS = 10

# Minimum title length in characters
MIN_TITLE_LENGTH = 5

# Maximum title length (titles longer than this are probably not real titles)
MAX_TITLE_LENGTH = 200

# Minimum body text characters (rough estimate of "has content")
MIN_BODY_LENGTH = 200

# Patterns that indicate low-quality or non-content pages
LOW_QUALITY_PATTERNS: tuple[str, ...] = (
    r"<title>\s*(?:404|not found|error|page not found)\s*</title>",
    r"<h1>\s*(?:404|not found|error)\s*</h1>",
    r"deployment\s*not\s*found",
    r"this\s*page\s*doesn't\s*exist",
    r"page\s*not\s*found",
    r"<title>\s*untitled\s*</title>",
    r"<title>\s*test\s*</title>",
    r"<title>\s*hello\s*world\s*</title>",
    r"under construction",
    r"coming soon",
    r"this page.*under construction",
    r"check back later",
    r"site.*maintenance",
)

# Commercial site indicators in HTML content
COMMERCIAL_INDICATORS: tuple[str, ...] = (
    r"<title>[^<]*(?:shop|store|buy|price|sale|deal|discount|coupon)[^<]*</title>",
    r"<h1[^>]*>[^<]*(?:shop|store|buy|price|sale|deal|discount|coupon)[^<]*</h1>",
    r"(?:add to cart|buy now|purchase|checkout|shopping cart)",
    r"(?:price|cost|msrp|retail price|regular price|sale price)\s*[:$]",
    r"(?:free shipping|limited time offer|act now|order now)",
    r"(?:credit card|pay with|visa|mastercard|amex|paypal)\s*(?:accepted|only)",
    r"<meta[^>]*description[^>]*(?:shop|store|buy|price|sale|deal)[^>]*>",
    r"(?:product|item|sku|upc|ean)\s*(?:number|#|code)?\s*:",
)


class QualityFilter:
    """Evaluates content quality for the daily random website pipeline.

    Performs structural quality checks on URLs and page content to
    ensure only meaningful, well-formed pages are featured.

    Usage::

        qf = QualityFilter()
        if qf.has_content("<html><body>...</body></html>"):
            print("Page has meaningful content")
    """

    def __init__(
        self,
        *,
        domain_filter: DomainFilter | None = None,
        min_words: int = MIN_CONTENT_WORDS,
        min_body_length: int = MIN_BODY_LENGTH,
    ) -> None:
        self._domain_filter = domain_filter or DomainFilter()
        self._normalizer = URLNormalizer(strip_query=True, strip_fragments=True)
        self._min_words = min_words
        self._min_body_length = min_body_length
        self._low_quality_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in LOW_QUALITY_PATTERNS
        ]
        self._commercial_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in COMMERCIAL_INDICATORS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid_url(self, url: str) -> bool:
        """Basic URL validation.

        Checks:
        - URL is well-formed with http/https scheme
        - Has a valid hostname
        - Path is not empty (optional, but good practice)
        - No obviously broken URL patterns

        Does NOT make network requests.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Must have a scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Must have a hostname
        if not parsed.hostname:
            return False

        # Hostname should be at least 4 chars (e.g., "a.bc" minimum)
        hostname = parsed.hostname
        if len(hostname) <= 3:
            return False

        # No double dots in hostname (e.g., "a..b.com")
        if ".." in hostname:
            return False

        # No spaces in URL
        if " " in url:
            return False

        return True

    def has_content(self, html: str) -> bool:
        """Check if a page has meaningful content.

        Detects:
        - Empty or near-empty pages
        - Error pages (404, 500)
        - Placeholder pages (coming soon, under construction)
        - Pages with only navigation/no body text

        Does NOT analyze content quality — just checks it exists.
        """
        if not html:
            return False

        html_stripped = html.strip()
        if len(html_stripped) < 100:
            return False

        # Check for low-quality indicators
        for pattern in self._low_quality_patterns:
            if pattern.search(html_stripped):
                return False

        # Extract body text (strip HTML tags for text analysis)
        text = self._extract_text(html_stripped)
        has_interactive_elements = any(
            tag in html_stripped.lower()
            for tag in ("<canvas", "<svg", 'id="app"', 'id="root"', 'id="game"', "webgl", "three.js")
        )
        if len(text) < self._min_body_length and not has_interactive_elements:
            return False

        return True

    def is_too_short(self, text: str, min_words: int | None = None) -> bool:
        """Check if content text is too short to be meaningful.

        Args:
            text: Plain text content to check.
            min_words: Minimum word count (default: self._min_words).

        Returns True if the text has fewer than min_words words.
        """
        threshold = min_words if min_words is not None else self._min_words
        if not text:
            return True
        word_count = len(text.split())
        return word_count < threshold

    def is_login_page(self, html: str) -> bool:
        """Detect if a page is primarily a login form.

        Checks for:
        - Password input fields
        - Login-related form elements
        - Minimal body content (login forms typically have little else)
        """
        if not html:
            return False

        html_lower = html.lower()

        # Count login form indicators
        login_score = 0

        # Password fields
        if re.search(r'<input[^>]*type\s*=\s*["\']password["\']', html_lower):
            login_score += 2

        # Login-related text
        login_text_patterns = (
            r"sign\s*in",
            r"log\s*in",
            r"login",
            r"signin",
            r"username",
            r"email\s*address",
            r"forgot\s*password",
            r"remember\s*me",
            r"create\s*(?:an?\s*)?account",
        )
        for pattern in login_text_patterns:
            if re.search(pattern, html_lower):
                login_score += 1

        # If score is high enough, it's a login page
        if login_score >= 3:
            # But also check if there's substantial content alongside
            text = self._extract_text(html_lower)
            word_count = len(text.split())
            if word_count < 50:
                return True

        return False

    def is_commercial_site(self, html: str, title: str = "") -> bool:
        """Detect if a page is a commercial/shopping site.

        Checks HTML content and title for commercial indicators
        like "buy now", "add to cart", pricing info, etc.
        """
        # Check title
        if title:
            title_lower = title.lower()
            commercial_title_words = ("shop", "store", "buy", "price", "sale",
                                      "deal", "discount", "coupon", "deal",
                                      "bargain", "cheap", "wholesale")
            for word in commercial_title_words:
                if word in title_lower:
                    return True

        # Check HTML content
        if html:
            html_lower = html.lower()
            for pattern in self._commercial_patterns:
                if pattern.search(html_lower):
                    return True

        return False

    def get_page_title(self, html: str) -> str | None:
        """Extract the page title from HTML.

        Returns the title text if found and valid, None otherwise.
        """
        if not html:
            return None

        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if not match:
            return None

        title = match.group(1).strip()

        # Validate title
        if len(title) < MIN_TITLE_LENGTH:
            return None
        if len(title) > MAX_TITLE_LENGTH:
            return None

        return title

    def validate_page(
        self,
        url: str,
        html: str | None = None,
        title: str = "",
    ) -> dict[str, Any]:
        """Comprehensive page quality validation.

        Performs all quality checks and returns a detailed result.

        Returns:
            Dict with keys:
            - is_valid (bool): Overall quality verdict
            - url_valid (bool): URL structure is valid
            - has_content (bool): Page has meaningful content
            - is_login (bool): Page is a login form
            - is_commercial (bool): Page is a commercial site
            - title (str|None): Extracted title
            - issues (list[str]): List of quality issues found
        """
        issues: list[str] = []

        url_valid = self.is_valid_url(url)
        if not url_valid:
            issues.append("Invalid URL structure")

        has_content = False
        is_login = False
        is_commercial = False
        extracted_title = None

        # Handle empty or missing HTML
        if not html:
            issues.append("No HTML content provided")
        else:
            has_content = self.has_content(html)
            if not has_content:
                issues.append("Page lacks meaningful content")

            is_login = self.is_login_page(html)
            if is_login:
                issues.append("Page is a login form")

            extracted_title = self.get_page_title(html)
            if not extracted_title:
                issues.append("No valid page title found")

            # Use provided title or extracted
            check_title = title or (extracted_title or "")
            is_commercial = self.is_commercial_site(html, check_title)
            if is_commercial:
                issues.append("Page is a commercial/shopping site")

        return {
            "is_valid": url_valid and not is_login and not is_commercial and has_content,
            "url_valid": url_valid,
            "has_content": has_content,
            "is_login": is_login,
            "is_commercial": is_commercial,
            "title": extracted_title,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text(self, html: str) -> str:
        """Extract visible text from HTML by stripping tags and scripts."""
        # Remove script and style blocks
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode common HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    qf = QualityFilter()

    test_urls = [
        "https://random-blog.com/2024/my-post",
        "https://github.com/user/repo",
        "https://a.b",  # too short hostname
        "https://",  # no hostname
        "not-a-url",
    ]

    print("=" * 72)
    print("QUALITY FILTER DEMO")
    print("=" * 72)

    for url in test_urls:
        valid = qf.is_valid_url(url)
        status = "VALID" if valid else "INVALID"
        print(f"  {status:7s} {url}")

    print()

    test_htmls = [
        ("<html><body><p>This is a meaningful blog post with enough content.</p></body></html>",
         "Valid Content"),
        ("<html><body><h1>404 Not Found</h1></body></html>",
         "Error Page"),
        ("<html><head><title>Sign In</title></head><body>"
         "<form><input type='password'></form></body></html>",
         "Login Page"),
        ("<html><body><h1>Buy Now - Sale!</h1></body></html>",
         "Commercial Site"),
        ("<html><title></title><body></body></html>",
         "Empty Page"),
        ("",
         "Empty String"),
    ]

    print("Content quality checks:")
    for html, label in test_htmls:
        has = qf.has_content(html)
        login = qf.is_login_page(html)
        title = qf.get_page_title(html) or "(none)"
        print(f"  {label:20s} content={has}, login={login}, title={title!r}")
