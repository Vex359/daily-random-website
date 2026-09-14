"""Domain filtering module.

Blocks known major corporations, commercial sites, unsafe content,
and login-gate domains from being featured as "random" discoveries.

Uses URLNormalizer from dedup.py for domain extraction.
"""

from __future__ import annotations

from src.filters.dedup import URLNormalizer


# ---------------------------------------------------------------------------
# Blocklists
# ---------------------------------------------------------------------------

# Major corporations — always blocked (wouldn't be "random" discoveries)
MAJOR_CORPORATIONS: frozenset[str] = frozenset({
    # Big Tech
    "google.com",
    "youtube.com",
    "facebook.com",
    "fb.com",
    "instagram.com",
    "whatsapp.com",
    "apple.com",
    "icloud.com",
    "microsoft.com",
    "live.com",
    "office.com",
    "office365.com",
    "amazon.com",
    "aws.amazon.com",
    "netflix.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "snapchat.com",
    "pinterest.com",
    "threads.net",
    "mastodon.social",
    # E-commerce / Payments
    "ebay.com",
    "paypal.com",
    "stripe.com",
    "shopify.com",
    "walmart.com",
    "target.com",
    "bestbuy.com",
    "costco.com",
    "etsy.com",
    "aliexpress.com",
    "alibaba.com",
    # Social / Communication
    "discord.com",
    "slack.com",
    "zoom.us",
    "teams.microsoft.com",
    "twitch.tv",
    "spotify.com",
    "soundcloud.com",
    # News / Media
    "cnn.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "washingtonpost.com",
    "reuters.com",
    "bloomberg.com",
    "forbes.com",
    "wsj.com",
    "huffpost.com",
    "foxnews.com",
    "msnbc.com",
    "usatoday.com",
    "theguardian.com",
    "dw.com",
    # Cloud / SaaS
    "cloudflare.com",
    "akamai.com",
    "vercel.com",
    "netlify.com",
    "heroku.com",
    "digitalocean.com",
    "dropbox.com",
    "box.com",
    "notion.so",
    "atlassian.com",
    "trello.com",
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    # Dev Tools / Platforms
    "stackoverflow.com",
    "stackexchange.com",
    "medium.com",
    "substack.com",
    "dev.to",
    "npmjs.com",
    "pypi.org",
    "docker.com",
    "dockerhub.com",
    # Auto / Transport
    "tesla.com",
    "uber.com",
    "lyft.com",
    "airbnb.com",
})

# Domains that are always unsafe — malware, phishing, gambling, adult
UNSAFE_DOMAINS: frozenset[str] = frozenset({
    # Malware / Phishing
    "malware.com",
    "malwarebytes.com",  # legit security, but not a discovery target
    "phishing.com",
    "virustotal.com",
    # Gambling
    "gambling.com",
    "casino.com",
    "poker.com",
    "bet365.com",
    "draftkings.com",
    "fanduel.com",
    "vegas.com",
    "bwin.com",
    "888.com",
    "888casino.com",
    # Adult content
    "pornhub.com",
    "xvideos.com",
    "xhamster.com",
    "redtube.com",
    "youporn.com",
    "onlyfans.com",
    "chaturbate.com",
    "livejasmin.com",
})

# Commercial pattern prefixes — these domain prefixes indicate a store
COMMERCIAL_PREFIXES: tuple[str, ...] = (
    "shop.",
    "store.",
    "buy.",
    "price.",
    "deal.",
    "sale.",
    "cheap.",
    "discount.",
    "coupon.",
    "bargain.",
    "wholesale.",
    "mall.",
)

# Login / auth domains — not interesting content
LOGIN_DOMAINS: frozenset[str] = frozenset({
    "accounts.google.com",
    "login.microsoftonline.com",
    "login.yahoo.com",
    "id.apple.com",
    "facebook.com/login",
    "twitter.com/login",
    "github.com/login",
    "gitlab.com/users/sign_in",
})

# Commercial TLD patterns (just TLDs that are rarely useful)
COMMERCIAL_TLDS: frozenset[str] = frozenset({
    ".shop",
    ".store",
    ".buy",
    ".sale",
    ".deals",
    ".coupons",
})

# Domain categories for classification
DOMAIN_CATEGORIES = {
    "major_corporation": "major_corporation",
    "commercial": "commercial",
    "unsafe": "unsafe",
    "login": "login",
    "news_media": "news_media",
    "technology": "technology",
    "education": "education",
    "government": "government",
    "social_media": "social_media",
    "unknown": "unknown",
}


class DomainFilter:
    """Filters URLs based on domain reputation and classification.

    Uses URLNormalizer from dedup.py for consistent domain extraction.
    Provides methods to check blocklists, classify domains, and filter
    URL lists before they enter the content pipeline.

    Usage::

        df = DomainFilter()
        if df.is_blocked("google.com"):
            print("Skipped: major corporation")
    """

    def __init__(
        self,
        *,
        extra_blocked: set[str] | None = None,
        extra_corporations: set[str] | None = None,
        extra_unsafe: set[str] | None = None,
    ) -> None:
        self._normalizer = URLNormalizer(strip_query=True, strip_fragments=True)

        # Merge built-in lists with user-supplied extras
        self._major_corporations = MAJOR_CORPORATIONS | (extra_corporations or set())
        self._unsafe_domains = UNSAFE_DOMAINS | (extra_unsafe or set())
        self._extra_blocked = extra_blocked or set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_blocked(self, domain: str) -> bool:
        """Check if a domain should be blocked from the pipeline.

        A domain is blocked if it matches any of:
        - Major corporation blocklist
        - Unsafe domain blocklist
        - Login/auth domain
        - Commercial domain (shop.*, store.*, buy.*, etc.)
        - Extra blocked domains provided at init
        """
        normalized = self._normalize_domain(domain)
        hostname = self._extract_hostname(domain)

        if normalized in self._major_corporations:
            return True
        if normalized in self._unsafe_domains:
            return True
        if normalized in self._extra_blocked:
            return True
        if self._is_login_domain(normalized):
            return True
        if self.is_commercial(hostname):
            return True
        return False

    def is_major_corporation(self, domain: str) -> bool:
        """Check if a domain belongs to a major corporation.

        Matches both exact domains and common subdomains
        (e.g., mail.google.com -> google.com).
        """
        normalized = self._normalize_domain(domain)
        root = self._normalizer.extract_domain(normalized)
        return root in self._major_corporations

    def is_commercial(self, domain: str) -> bool:
        """Check if a domain is obviously commercial.

        Checks:
        - Domain has commercial prefix (shop.*, store.*, buy.*, etc.)
        - Domain TLD is a commercial TLD (.shop, .store, etc.)
        """
        normalized = self._normalize_domain(domain)
        full_domain = normalized  # e.g. "shop.example.com"

        # Check commercial prefixes
        for prefix in COMMERCIAL_PREFIXES:
            if full_domain.startswith(prefix):
                return True

        # Check commercial TLDs
        for tld in COMMERCIAL_TLDS:
            if normalized.endswith(tld):
                return True

        return False

    def get_domain_category(self, domain: str) -> str:
        """Classify a domain into a category.

        Returns one of the DOMAIN_CATEGORIES values:
        - major_corporation, commercial, unsafe, login
        - news_media, technology, education, government, social_media
        - unknown
        """
        normalized = self._normalize_domain(domain)
        root = self._normalizer.extract_domain(normalized)

        if root in self._major_corporations or normalized in self._major_corporations:
            return "major_corporation"
        if root in self._unsafe_domains or normalized in self._unsafe_domains:
            return "unsafe"
        if self._is_login_domain(normalized):
            return "login"
        if self.is_commercial(domain):
            return "commercial"

        # Heuristic classification for common domain patterns
        category = self._heuristic_category(normalized, root)
        if category:
            return category

        return "unknown"

    def filter_urls(self, urls: list[str]) -> list[dict[str, str]]:
        """Filter a list of URLs, removing blocked domains.

        Returns list of dicts with keys: url, canonical_url, domain, category.
        Only includes URLs that pass all filters.
        """
        results: list[dict[str, str]] = []
        for url in urls:
            entry = self._normalizer.normalize_urls([url])
            if not entry:
                continue
            domain = entry[0]["domain"]
            if self.is_blocked(domain):
                continue
            category = self.get_domain_category(domain)
            results.append({
                "url": url,
                "canonical_url": entry[0]["canonical_url"],
                "domain": domain,
                "category": category,
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_domain(self, domain: str) -> str:
        """Normalize a domain string for consistent comparison."""
        d = domain.lower().strip()
        if d.startswith("https://") or d.startswith("http://"):
            d = self._normalizer.extract_domain(d)
        if d.startswith("www."):
            d = d[4:]
        return d

    def _extract_hostname(self, url_or_domain: str) -> str:
        """Extract the full hostname (including subdomains) from a URL or domain string.

        Unlike _normalize_domain which extracts the root domain for blocklist matching,
        this preserves the full hostname for prefix-based checks like commercial detection.
        """
        from urllib.parse import urlparse

        d = url_or_domain.lower().strip()
        if "://" in d:
            parsed = urlparse(d)
            hostname = parsed.hostname or ""
        else:
            hostname = d.split("/")[0]
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname

    def _is_login_domain(self, domain: str) -> bool:
        """Check if a domain is a login/auth page."""
        if domain in LOGIN_DOMAINS:
            return True
        # Check for common login prefixes
        login_prefixes = ("accounts.", "login.", "auth.", "signin.", "sso.")
        for prefix in login_prefixes:
            if domain.startswith(prefix):
                return True
        return False

    def _heuristic_category(self, domain: str, root: str) -> str | None:
        """Apply heuristic rules to classify unknown domains."""
        # Government (check first — .gov TLD is definitive)
        if root.endswith(".gov") or domain.endswith(".gov") or ".gov." in domain:
            return "government"
        gov_keywords = ("government", "parliament", "congress", "council")
        for kw in gov_keywords:
            if kw in domain:
                return "government"

        # Education (check before technology — "learn-python" is education, not tech)
        edu_keywords = ("edu", "university", "college", "school", "learn",
                        "academy", "course", "study", "research")
        for kw in edu_keywords:
            if kw in domain:
                return "education"

        # News / Media
        news_keywords = ("news", "times", "post", "journal", "tribune", "herald",
                         "gazette", "chronicle", "observer", "mirror", "express",
                         "daily", "weekly", "magazine", "press")
        for kw in news_keywords:
            if kw in domain:
                return "news_media"

        # Technology (use specific compound keywords, not single chars like "net" or "io")
        tech_keywords = ("tech", "code", "dev", "hack", "byte", "cloud",
                         "software", "git", "golang")
        for kw in tech_keywords:
            if kw in root:
                return "technology"

        # Social Media
        social_keywords = ("social", "forum", "community", "chat", "board")
        for kw in social_keywords:
            if kw in domain:
                return "social_media"

        return None


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    df = DomainFilter()

    test_urls = [
        "https://www.google.com/search?q=test",
        "https://github.com/user/repo",
        "https://shop.example.com/product",
        "https://casino.com/slots",
        "https://random-indie-blog.com/2024/interesting-post",
        "https://news.ycombinator.com/item?id=123",
        "https://mail.yahoo.com/inbox",
        "https://buy-cheap-stuff.store/deals",
        "https://accounts.google.com/signin",
        "https://docs.python.org/3/tutorial",
        "https://unknown-personal-site.net/about",
    ]

    print("=" * 72)
    print("DOMAIN FILTER DEMO")
    print("=" * 72)

    for url in test_urls:
        entry = df.filter_urls([url])
        if entry:
            e = entry[0]
            print(f"  PASS  {e['domain']:30s} [{e['category']:20s}] {url}")
        else:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = (parsed.hostname or "").lower()
            if domain.startswith("www."):
                domain = domain[4:]
            category = df.get_domain_category(domain)
            print(f"  BLOCK {domain:30s} [{category:20s}] {url}")

    print()
    print(f"Total blocked domains: {len(df._major_corporations) + len(df._unsafe_domains)}")
