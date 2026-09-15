"""Post storage and persistence manager.

Manages reading/writing of post data, tracks seen domains,
and handles the local JSON database for the pipeline.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Data directory relative to project root (two levels up from src/storage/)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_POSTS_FILE = _DATA_DIR / "posts.json"
_SEEN_DOMAINS_FILE = _DATA_DIR / "seen_domains.json"

# Valid categories for posts
VALID_CATEGORIES: frozenset[str] = frozenset({
    "Interactive",
    "Weird",
    "Tools",
    "Games",
    "Art",
    "Educational",
    "General",
})

# Valid source types
VALID_SOURCES: frozenset[str] = frozenset({
    "hackernews",
    "github",
    "rss",
    "wikipedia",
    "awesome_lists",
})


class StorageManager:
    """Manages post storage in local JSON files.

    Provides methods for loading, saving, querying, and cleaning up
    posts. Tracks seen domains to prevent duplicates across pipeline runs.

    All methods that return posts return new lists — the internal state
    is never mutated in place by callers.
    """

    def __init__(
        self,
        *,
        posts_path: Path | str | None = None,
        seen_domains_path: Path | str | None = None,
        default_keep_count: int = 50,
    ) -> None:
        self._posts_path = Path(posts_path) if posts_path else _POSTS_FILE
        self._seen_domains_path = Path(seen_domains_path) if seen_domains_path else _SEEN_DOMAINS_FILE
        self._default_keep_count = default_keep_count
        self._posts: list[dict[str, Any]] = self._load_posts()
        self._seen_domains: set[str] = set(self._load_seen_domains())
        for p in self._posts:
            d = p.get("domain")
            if d:
                self._seen_domains.add(d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_posts(self) -> list[dict[str, Any]]:
        """Load all posts from the JSON file.

        Returns a new list copy — callers cannot mutate internal state.
        Returns an empty list if the file is missing or corrupt.
        """
        self._posts = self._load_posts()
        return list(self._posts)

    def save_posts(self, posts: list[dict[str, Any]]) -> None:
        """Save posts to the JSON file.

        Replaces the entire post list. Does not modify the input list.
        Updates the in-memory cache.
        """
        self._posts = list(posts)
        self._persist_posts()

    def add_post(self, post: dict[str, Any]) -> dict[str, Any]:
        """Add a new post.

        Assigns an id and discovered_at timestamp if not present.
        Adds the post's domain to seen domains.
        Returns the post with assigned fields.

        Raises:
            ValueError: If the domain already exists in seen domains.
        """
        post = dict(post)  # don't mutate caller's dict

        # Generate id if missing
        if not post.get("id"):
            post["id"] = str(uuid.uuid4())

        # Set discovered_at if missing
        if not post.get("discovered_at"):
            post["discovered_at"] = datetime.now(timezone.utc).isoformat()

        # Validate domain is not a duplicate
        domain = post.get("domain", "")
        if domain and self.domain_exists(domain):
            raise ValueError(f"Domain already exists: {domain}")

        # Ensure domain is tracked
        if domain:
            self._seen_domains.add(domain)

        self._posts.append(post)
        self._persist_posts()
        self._persist_seen_domains()
        return post

    def get_post(self, post_id: str) -> dict[str, Any] | None:
        """Get a post by its ID.

        Returns None if no post with the given ID exists.
        """
        for post in self._posts:
            if post.get("id") == post_id:
                return dict(post)
        return None

    def get_latest_posts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most recent posts, sorted by discovered_at descending.

        Returns up to `limit` posts. Returns a new list.
        """
        sorted_posts = sorted(
            self._posts,
            key=lambda p: p.get("discovered_at", ""),
            reverse=True,
        )
        return [dict(p) for p in sorted_posts[:limit]]

    def get_posts_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get all posts in a given category.

        Returns a new list of matching posts.
        """
        return [
            dict(p) for p in self._posts
            if p.get("category") == category
        ]

    def domain_exists(self, domain: str) -> bool:
        """Check if a domain has already been published.

        Checks both the in-memory set and the posts list.
        """
        if domain in self._seen_domains:
            return True
        return any(p.get("domain") == domain for p in self._posts)

    def cleanup_old_posts(self, keep_count: int | None = None) -> list[dict[str, Any]]:
        """Remove old posts, keeping the most recent ones.

        Args:
            keep_count: Number of posts to keep. Defaults to
                `default_keep_count` set at init.

        Returns:
            List of archived (removed) posts.
        """
        if keep_count is None:
            keep_count = self._default_keep_count

        if len(self._posts) <= keep_count:
            return []

        # Sort by discovered_at descending (newest first)
        sorted_posts = sorted(
            self._posts,
            key=lambda p: p.get("discovered_at", ""),
            reverse=True,
        )

        kept = sorted_posts[:keep_count]
        archived = sorted_posts[keep_count:]

        self._posts = kept
        self._persist_posts()

        return [dict(p) for p in archived]

    def get_seen_domains(self) -> set[str]:
        """Return the current set of seen domains."""
        return set(self._seen_domains)

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load_posts(self) -> list[dict[str, Any]]:
        """Load posts from the JSON file."""
        if not self._posts_path.exists():
            return []
        try:
            data = json.loads(self._posts_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def _persist_posts(self) -> None:
        """Write posts to the JSON file."""
        self._posts_path.parent.mkdir(parents=True, exist_ok=True)
        self._posts_path.write_text(
            json.dumps(self._posts, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _load_seen_domains(self) -> list[str]:
        """Load seen domains from the JSON file."""
        if not self._seen_domains_path.exists():
            return []
        try:
            data = json.loads(self._seen_domains_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError):
            return []

    def _persist_seen_domains(self) -> None:
        """Write seen domains to the JSON file."""
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
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manager = StorageManager(
            posts_path=tmp / "posts.json",
            seen_domains_path=tmp / "seen_domains.json",
        )

        print("=" * 72)
        print("STORAGE MANAGER DEMO")
        print("=" * 72)

        # Add some posts
        post1 = manager.add_post({
            "url": "https://cool-site.com/page",
            "canonical_url": "https://cool-site.com/page",
            "domain": "cool-site.com",
            "title": "Cool Site",
            "description": "A really cool site",
            "category": "Interactive",
            "source": "hackernews",
            "source_type": "show_hn",
            "score": 85,
            "screenshot_path": "screenshots/cool-site-com.webp",
            "ai_description": "Check out this amazing interactive website!",
            "why_interesting": "Innovative use of web APIs",
        })
        print(f"\nAdded post: {post1['id'][:8]}... ({post1['domain']})")

        post2 = manager.add_post({
            "url": "https://weird-thing.org",
            "canonical_url": "https://weird-thing.org",
            "domain": "weird-thing.org",
            "title": "Weird Thing",
            "description": "Something strange",
            "category": "Weird",
            "source": "rss",
            "source_type": "creative_coding",
            "score": 72,
            "screenshot_path": "screenshots/weird-thing-org.webp",
            "ai_description": "This website does something you've never seen.",
            "why_interesting": "Unique creative approach",
        })
        print(f"Added post: {post2['id'][:8]}... ({post2['domain']})")

        # Domain check
        print(f"\nDomain 'cool-site.com' exists: {manager.domain_exists('cool-site.com')}")
        print(f"Domain 'unknown.com' exists:   {manager.domain_exists('unknown.com')}")

        # Duplicate domain
        try:
            manager.add_post({"domain": "cool-site.com"})
        except ValueError as e:
            print(f"\nDuplicate rejected: {e}")

        # Get post
        found = manager.get_post(post1["id"])
        print(f"\nGet post by ID: {found['title'] if found else 'Not found'}")

        # Latest posts
        latest = manager.get_latest_posts(limit=5)
        print(f"\nLatest posts ({len(latest)}):")
        for p in latest:
            print(f"  - {p['title']} ({p['category']})")

        # By category
        weird = manager.get_posts_by_category("Weird")
        print(f"\nWeird posts: {len(weird)}")

        # All posts
        all_posts = manager.load_posts()
        print(f"\nTotal posts: {len(all_posts)}")

        # Cleanup
        archived = manager.cleanup_old_posts(keep_count=1)
        print(f"\nArchived {len(archived)} posts")
        print(f"Remaining: {len(manager.load_posts())}")

        # Verify JSON is valid
        raw = json.loads((tmp / "posts.json").read_text(encoding="utf-8"))
        print(f"\nJSON valid: {isinstance(raw, list)}")
        print(f"Posts in file: {len(raw)}")
