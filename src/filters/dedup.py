"""Content deduplication filter.

Removes duplicate and near-duplicate content using URL normalization,
title similarity, and content fingerprinting.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse

_MULTI_PART_TLDS: frozenset[str] = frozenset({
    "co.uk", "co.jp", "co.kr", "co.nz", "co.za", "co.in", "co.id",
    "com.au", "com.br", "com.cn", "com.mx", "com.sg", "com.tw",
    "org.uk", "net.au", "gov.uk", "ac.uk",
    "com.ar", "com.co", "org.au", "net.nz",
    "or.jp", "ne.jp", "go.jp",
})

# Data directory relative to project root (two levels up from src/filters/)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_SEEN_DOMAINS_FILE = _DATA_DIR / "seen_domains.json"


class URLNormalizer:
    """Normalizes URLs and removes duplicates for the content pipeline.

    Maintains both the original URL and a canonical (normalized) form.
    Optionally persists seen domains across runs via a JSON file.
    """

    def __init__(
        self,
        *,
        strip_query: bool = True,
        strip_fragments: bool = True,
        seen_domains_path: Path | str | None = None,
    ) -> None:
        self.strip_query = strip_query
        self.strip_fragments = strip_fragments
        self._seen_domains_path = Path(seen_domains_path) if seen_domains_path else _SEEN_DOMAINS_FILE
        self._seen_domains: set[str] = set(self._load_seen_domains())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_url(self, url: str) -> str:
        """Return the canonical form of a single URL.

        Rules applied (in order):
        1. Convert http:// to https://
        2. Lowercase the hostname
        3. Remove www. prefix
        4. Remove default ports (443 or 80)
        5. Remove trailing slashes
        6. Remove /index.html / /index.htm
        7. Remove fragment (#...)
        8. Optionally remove query parameters
        """
        parsed = urlparse(url)

        scheme = "https"

        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]

        port = parsed.port
        if port in (443, 80):
            port = None

        netloc = hostname
        if port is not None:
            netloc = f"{hostname}:{port}"

        path = parsed.path
        if path:
            path = path.lower()
            if path.endswith("/index.html"):
                path = path[: -len("/index.html")]
            elif path.endswith("/index.htm"):
                path = path[: -len("/index.htm")]
            if path == "/":
                path = ""
            elif len(path) > 1 and path.endswith("/"):
                path = path.rstrip("/")
        elif not netloc:
            path = "/"

        fragment = "" if self.strip_fragments else parsed.fragment

        if self.strip_query:
            query = ""
        else:
            query = parsed.query

        rebuilt = urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            query,
            fragment,
        ))

        return rebuilt

    def extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL.

        Handles subdomains and multi-part TLDs:
            blog.example.com          -> example.com
            www.news.example.co.uk    -> example.co.uk
            deep.sub.example.com      -> example.com
        """
        parsed = urlparse(url if "://" in url else f"https://{url}")
        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]

        parts = hostname.split(".")
        if len(parts) <= 2:
            return hostname

        for i in range(1, len(parts)):
            candidate_tld = ".".join(parts[i:])
            if candidate_tld in _MULTI_PART_TLDS:
                return ".".join(parts[i - 1 :])

        return ".".join(parts[-2:])

    def normalize_urls(self, urls: list[str]) -> list[dict[str, str]]:
        """Normalize a list of URLs.

        Returns a list of dicts with keys: url, canonical_url, domain.
        """
        results: list[dict[str, str]] = []
        for url in urls:
            canonical = self.normalize_url(url)
            domain = self.extract_domain(canonical)
            results.append({
                "url": url,
                "canonical_url": canonical,
                "domain": domain,
            })
        return results

    def deduplicate(self, urls: list[str]) -> list[dict[str, str]]:
        """Remove duplicate URLs and domains from a list.

        Deduplication strategy:
        1. Exact canonical-URL duplicates removed.
        2. Domain duplicates removed (keeps first occurrence).
        3. Previously seen domains (from disk) also filtered out.

        Returns a list of dicts with keys: url, canonical_url, domain.
        """
        normalized = self.normalize_urls(urls)

        seen_canonicals: set[str] = set()
        seen_domains: set[str] = set()
        unique: list[dict[str, str]] = []

        for entry in normalized:
            canonical = entry["canonical_url"]
            domain = entry["domain"]

            if canonical in seen_canonicals:
                continue

            if domain in seen_domains:
                continue

            if domain in self._seen_domains:
                continue

            seen_canonicals.add(canonical)
            seen_domains.add(domain)
            unique.append(entry)

        new_domains = seen_domains - self._seen_domains
        if new_domains:
            self._seen_domains.update(new_domains)
            self._save_seen_domains()

        return unique

    def get_seen_domains(self) -> set[str]:
        """Return the current set of previously seen domains."""
        return set(self._seen_domains)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_seen_domains(self) -> list[str]:
        """Load previously seen domains from the JSON file."""
        if not self._seen_domains_path.exists():
            return []
        try:
            data = json.loads(self._seen_domains_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_seen_domains(self) -> None:
        """Persist seen domains to the JSON file."""
        self._seen_domains_path.parent.mkdir(parents=True, exist_ok=True)
        data = sorted(self._seen_domains)
        self._seen_domains_path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )


# ------------------------------------------------------------------
# CLI demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    demo_urls = [
        "http://www.example.com/",
        "https://example.com",
        "https://Example.COM/page/index.html",
        "https://blog.example.co.uk/post?ref=home#top",
        "http://news.example.co.uk/article?q=1",
        "https://www.example.com/page",
        "https://example.com/page/",
        "https://subdomain.example.com/deep/path",
        "http://example.com:443/secure",
        "https://example.com:80/insecure",
    ]

    normalizer = URLNormalizer(strip_query=True)

    print("=" * 72)
    print("URL NORMALIZATION DEMO")
    print("=" * 72)

    results = normalizer.normalize_urls(demo_urls)
    for entry in results:
        print(f"  Original:  {entry['url']}")
        print(f"  Canonical: {entry['canonical_url']}")
        print(f"  Domain:    {entry['domain']}")
        print()

    print("-" * 72)
    print("DEDUPLICATION")
    print("-" * 72)

    deduped = normalizer.deduplicate(demo_urls)
    print(f"  Input URLs:  {len(demo_urls)}")
    print(f"  After dedup: {len(deduped)}")
    print()
    for entry in deduped:
        print(f"  {entry['domain']:30s} -> {entry['canonical_url']}")

    print()
    print(f"Seen domains file: {normalizer._seen_domains_path}")
    print(f"Tracked domains:   {normalizer.get_seen_domains()}")
