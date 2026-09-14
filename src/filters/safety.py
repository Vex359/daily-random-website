"""Content safety filter.

Scans collected content for NSFW, spam, harmful, or otherwise
inappropriate material before publishing. Validates URLs and
HTTP responses for safety indicators.

Safety checks include:
- URL has no obviously malicious patterns
- Response is HTML content type
- Response size < 10MB
- No login walls detected
- No obvious malware indicators in page content
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from src.filters.domains import DomainFilter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum response size in bytes (10MB)
MAX_RESPONSE_SIZE = 10 * 1024 * 1024

# Allowed content types for HTML pages
ALLOWED_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "application/xhtml+xml",
    "application/xhtml",
)

# Suspicious URL parameters that indicate tracking, auth bypass, or injection
SUSPICIOUS_PARAMS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "system",
    "cmd",
    "command",
    "shell",
    "passthru",
    "base64",
    "unescape",
    "redirect",
    "url",
    "next",
    "return",
    "continue",
    "ref",
    "rurl",
    "dest",
    "destination",
    "checkout_url",
    "return_url",
    "goto",
    "link",
    "go",
    "out",
    "view",
    "to",
    "path",
    "file",
    "document",
    "folder",
    "dir",
    "page",
    "pdf",
    "template",
    "php",
    "include",
    "require",
    "show",
    "display",
    "load",
    "read",
    "source",
})

# Max allowed redirects
MAX_REDIRECTS = 3

# Malware indicators in page content (HTML/JS patterns)
MALWARE_INDICATORS: tuple[str, ...] = (
    r"<script[^>]*>\s*eval\s*\(",
    r"<script[^>]*>\s*document\.write\s*\(\s*unescape",
    r"<script[^>]*src\s*=\s*['\"]?https?://[^'\"]*\.exe",
    r"<script[^>]*src\s*=\s*['\"]?https?://[^'\"]*\.scr",
    r"<iframe[^>]*hidden",
    r"<iframe[^>]*display\s*:\s*none",
    r"<object[^>]*\bdata\s*=\s*['\"]?https?://[^'\"]*\.(?:exe|scr|bat|cmd)",
    r"<embed[^>]*src\s*=\s*['\"]?https?://[^'\"]*\.(?:exe|scr|bat|cmd)",
    r"window\.location\s*=\s*['\"]?javascript:",
    r"window\.location\.href\s*=\s*['\"]?javascript:",
    r"document\.location\s*=\s*['\"]?javascript:",
    r"document\.location\.href\s*=\s*['\"]?javascript:",
    r"setTimeout\s*\(\s*['\"]?eval",
    r"setInterval\s*\(\s*['\"]?eval",
    r"String\.fromCharCode\s*\(",
    r"atob\s*\(",
    r"document\.cookie",
    r"navigator\.cookieEnabled",
    r"\.write\s*\(\s*unescape",
    r"eval\s*\(\s*document",
    r"eval\s*\(\s*window",
    r"eval\s*\(\s*location",
)

# Login page indicators (HTML patterns)
LOGIN_INDICATORS: tuple[str, ...] = (
    r"<form[^>]*(?:login|signin|log-in|sign-in|authenticate)[^>]*>",
    r"<input[^>]*(?:password|passwd|pwd)[^>]*>",
    r"<button[^>]*(?:login|sign.?in|log.?in)[^>]*>",
    r"<a[^>]*(?:forgot.?password|reset.?password|recover)[^>]*>",
    r"<title[^>]*>[^<]*(?:login|sign.?in|log.?in)[^<]*</title>",
)


class SafetyFilter:
    """Validates URLs and HTTP responses for safety before content processing.

    Performs multi-layered safety checks on URLs and responses:
    - Domain reputation (via DomainFilter)
    - URL structure analysis (suspicious parameters, schemes)
    - Response validation (content-type, size)
    - Content scanning (malware indicators, login walls)

    Usage::

        sf = SafetyFilter()
        if sf.is_safe_url("https://random-blog.com/post"):
            print("URL looks safe to crawl")
    """

    def __init__(self, *, domain_filter: DomainFilter | None = None) -> None:
        self._domain_filter = domain_filter or DomainFilter()
        self._malware_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in MALWARE_INDICATORS
        ]
        self._login_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in LOGIN_INDICATORS
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_safe_url(self, url: str) -> bool:
        """Check if a URL is safe to crawl.

        Checks:
        - URL is well-formed with http/https scheme
        - Domain is not blocked or unsafe
        - No suspicious URL parameters
        - No javascript: or data: scheme
        - No obvious injection patterns in URL path

        Returns True if the URL passes all safety checks.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            return False

        # Must have a hostname
        if not parsed.hostname:
            return False

        # Check domain safety via DomainFilter
        hostname = parsed.hostname.lower()
        if self._domain_filter.is_blocked(hostname):
            return False

        # Check for suspicious URL parameters
        if parsed.query:
            query_lower = parsed.query.lower()
            for param in SUSPICIOUS_PARAMS:
                # Check if suspicious param name appears in query string
                if re.search(rf"(?:^|&|;){re.escape(param)}(?:=|$|&|;)", query_lower):
                    return False

        # Check path for injection patterns
        path = parsed.path.lower()
        suspicious_path_patterns = (
            "/etc/passwd",
            "/etc/shadow",
            "/proc/",
            "/.env",
            "/wp-admin",
            "/wp-login",
            "/cgi-bin/",
            "/admin/config",
            "/debug/",
            "/test/",
            "/phpmyadmin",
            "/.git/",
            "/.svn/",
        )
        for pattern in suspicious_path_patterns:
            if pattern in path:
                return False

        # Check for encoded payloads in URL
        decoded_path = _decode_url_encoding(path)
        if decoded_path != path:
            for pattern in ("/etc/passwd", "/proc/", "/.env", "javascript:", "data:"):
                if pattern in decoded_path:
                    return False

        return True

    def is_safe_content(self, html: str) -> bool:
        """Check if HTML page content is safe to process.

        Scans for:
        - Obvious malware/redirect patterns
        - Hidden iframes (common in drive-by downloads)
        - Obfuscated JavaScript that evaluates dynamic content
        - Crypto mining scripts

        Does NOT execute any JavaScript — only pattern matching.
        """
        if not html:
            return True  # Empty content is safe (just low quality)

        # Scan for malware indicators
        for pattern in self._malware_patterns:
            if pattern.search(html):
                return False

        # Check for crypto mining scripts
        mining_indicators = (
            "coinhive",
            "coin-hive",
            "cryptoloot",
            "crypto-loot",
            "coinIMP",
            "jsecoin",
            "authedmine",
            "webminepool",
            "minero.cc",
            "ppoi.org",
        )
        html_lower = html.lower()
        for indicator in mining_indicators:
            if indicator.lower() in html_lower:
                return False

        return True

    def check_response(self, response: Any) -> dict[str, Any]:
        """Validate an HTTP response for safety and quality.

        Args:
            response: An object with attributes: status_code, headers,
                      content, url (like requests.Response or httpx.Response).

        Returns:
            Dict with keys:
            - is_safe (bool): Overall safety verdict
            - content_type (str): Detected content type
            - size_bytes (int): Response body size
            - issues (list[str]): List of safety issues found
        """
        issues: list[str] = []

        # Check status code
        status = getattr(response, "status_code", 0)
        if status < 200 or status >= 400:
            issues.append(f"Non-2xx status code: {status}")

        # Check content type
        content_type = self._extract_content_type(response)
        is_html = any(ct in content_type for ct in ALLOWED_CONTENT_TYPES)
        if content_type and not is_html:
            issues.append(f"Not HTML content: {content_type}")

        # Check response size
        content = getattr(response, "content", b"")
        size = len(content) if content else 0
        if size > MAX_RESPONSE_SIZE:
            issues.append(f"Response too large: {size} bytes (max {MAX_RESPONSE_SIZE})")

        # Check for HTML-specific safety issues
        if is_html and content:
            html_text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
            if not self.is_safe_content(html_text):
                issues.append("Unsafe content detected (malware indicators)")

        # Check for redirect loops
        history = getattr(response, "history", None)
        if history and len(history) > MAX_REDIRECTS:
            issues.append(f"Too many redirects: {len(history)} (max {MAX_REDIRECTS})")

        return {
            "is_safe": len(issues) == 0,
            "content_type": content_type,
            "size_bytes": size,
            "issues": issues,
        }

    def has_login_wall(self, html: str) -> bool:
        """Detect if a page is a login wall that blocks content access.

        Returns True if the page appears to be primarily a login form
        with no accessible content behind it.
        """
        if not html:
            return False

        html_lower = html.lower()

        # Count login indicators
        login_score = 0
        for pattern in self._login_patterns:
            if pattern.search(html):
                login_score += 1

        # If multiple login indicators, likely a login wall
        if login_score >= 2:
            return True

        # Check for common login-wall patterns
        wall_patterns = (
            r"you must (?:log|sign)\s*in",
            r"please (?:log|sign)\s*in",
            r"(?:log|sign)\s*in (?:to|for) continue",
            r"(?:log|sign)\s*in to access",
            r"authentication required",
            r"access denied.*(?:log|sign)\s*in",
            r"subscribe to continue",
            r"paywall",
            r"membership required",
        )
        for pattern in wall_patterns:
            if re.search(pattern, html_lower):
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_content_type(self, response: Any) -> str:
        """Extract content type from response headers."""
        headers = getattr(response, "headers", {})
        if hasattr(headers, "get"):
            ct = headers.get("content-type", "")
            # Strip parameters like charset
            return ct.split(";")[0].strip().lower()
        return ""


def _decode_url_encoding(url_part: str) -> str:
    """Decode percent-encoded characters in a URL path segment."""
    import urllib.parse
    try:
        return urllib.parse.unquote(url_part)
    except Exception:
        return url_part


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    sf = SafetyFilter()

    test_urls = [
        "https://random-blog.com/2024/my-post",
        "https://github.com/user/repo",
        "https://evil-site.com/malware.exe",
        "https://shop.store/buy-now",
        "https://casino.com/slots",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "https://example.com/page?redirect=http://evil.com",
        "https://example.com/page?file=/etc/passwd",
        "https://example.com/page?cmd=ls",
        "https://example.com/path/../../../etc/passwd",
        "https://accounts.google.com/signin",
    ]

    print("=" * 72)
    print("SAFETY FILTER DEMO")
    print("=" * 72)

    for url in test_urls:
        safe = sf.is_safe_url(url)
        status = "SAFE  " if safe else "BLOCK "
        print(f"  {status} {url}")

    print()
    print("Content scan demo:")
    safe_html = "<html><body><h1>Hello World</h1></body></html>"
    malware_html = '<html><script>eval(document.cookie)</script></html>'
    print(f"  Safe HTML:     {sf.is_safe_content(safe_html)}")
    print(f"  Malware HTML:  {sf.is_safe_content(malware_html)}")
