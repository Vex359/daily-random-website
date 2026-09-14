"""Tests for storage manager.

Verifies post persistence, domain tracking, and data integrity
in the local JSON storage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.storage.manager import StorageManager, VALID_CATEGORIES


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def manager(tmp_path: Path) -> StorageManager:
    """Create a StorageManager with temp files."""
    return StorageManager(
        posts_path=tmp_path / "posts.json",
        seen_domains_path=tmp_path / "seen.json",
    )


@pytest.fixture()
def sample_post() -> dict:
    """A minimal valid post dict."""
    return {
        "url": "https://example.com/page",
        "canonical_url": "https://example.com/page",
        "domain": "example.com",
        "title": "Example Site",
        "description": "An example website",
        "category": "Interactive",
        "source": "hackernews",
        "source_type": "show_hn",
        "score": 85,
        "screenshot_path": "screenshots/example-com.webp",
        "ai_description": "Check out this example!",
        "why_interesting": "Great example of web design",
    }


@pytest.fixture()
def manager_with_posts(manager: StorageManager, sample_post: dict) -> StorageManager:
    """Pre-populate manager with several posts."""
    posts = [
        {**sample_post, "domain": "site1.com", "title": "Site 1", "category": "Interactive",
         "discovered_at": "2024-01-01T00:00:00Z"},
        {**sample_post, "domain": "site2.com", "title": "Site 2", "category": "Weird",
         "discovered_at": "2024-01-02T00:00:00Z"},
        {**sample_post, "domain": "site3.com", "title": "Site 3", "category": "Tools",
         "discovered_at": "2024-01-03T00:00:00Z"},
        {**sample_post, "domain": "site4.com", "title": "Site 4", "category": "Interactive",
         "discovered_at": "2024-01-04T00:00:00Z"},
        {**sample_post, "domain": "site5.com", "title": "Site 5", "category": "Weird",
         "discovered_at": "2024-01-05T00:00:00Z"},
    ]
    for p in posts:
        manager.add_post(p)
    return manager


# =========================================================================
# __init__ / constructor tests
# =========================================================================

class TestInit:

    def test_creates_empty_state(self, manager: StorageManager) -> None:
        assert manager.load_posts() == []

    def test_creates_parent_dirs_on_write(self, tmp_path: Path, sample_post: dict) -> None:
        nested = tmp_path / "deep" / "nested" / "posts.json"
        m = StorageManager(posts_path=nested, seen_domains_path=tmp_path / "seen.json")
        m.add_post(sample_post)
        assert nested.parent.exists()

    def test_default_keep_count(self, manager: StorageManager) -> None:
        assert manager._default_keep_count == 50

    def test_custom_keep_count(self, tmp_path: Path) -> None:
        m = StorageManager(
            posts_path=tmp_path / "posts.json",
            seen_domains_path=tmp_path / "seen.json",
            default_keep_count=10,
        )
        assert m._default_keep_count == 10


# =========================================================================
# load_posts tests
# =========================================================================

class TestLoadPosts:

    def test_load_empty(self, manager: StorageManager) -> None:
        posts = manager.load_posts()
        assert posts == []

    def test_load_returns_new_list(self, manager: StorageManager) -> None:
        a = manager.load_posts()
        b = manager.load_posts()
        assert a is not b
        assert a == b

    def test_load_after_add(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        loaded = manager.load_posts()
        assert len(loaded) == 1
        assert loaded[0]["domain"] == "example.com"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        m = StorageManager(
            posts_path=tmp_path / "nonexistent.json",
            seen_domains_path=tmp_path / "seen.json",
        )
        assert m.load_posts() == []

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "posts.json"
        bad_file.write_text("NOT JSON!!!", encoding="utf-8")
        m = StorageManager(posts_path=bad_file, seen_domains_path=tmp_path / "seen.json")
        assert m.load_posts() == []

    def test_load_non_list_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "posts.json"
        bad_file.write_text('{"not": "a list"}', encoding="utf-8")
        m = StorageManager(posts_path=bad_file, seen_domains_path=tmp_path / "seen.json")
        assert m.load_posts() == []


# =========================================================================
# save_posts tests
# =========================================================================

class TestSavePosts:

    def test_save_empty(self, manager: StorageManager) -> None:
        manager.save_posts([])
        assert manager.load_posts() == []

    def test_save_and_load(self, manager: StorageManager, sample_post: dict) -> None:
        manager.save_posts([sample_post])
        loaded = manager.load_posts()
        assert len(loaded) == 1
        assert loaded[0]["domain"] == "example.com"

    def test_save_replaces_all(self, manager: StorageManager, sample_post: dict) -> None:
        manager.save_posts([sample_post])
        manager.save_posts([{**sample_post, "domain": "other.com"}])
        loaded = manager.load_posts()
        assert len(loaded) == 1
        assert loaded[0]["domain"] == "other.com"

    def test_save_does_not_mutate_input(self, manager: StorageManager, sample_post: dict) -> None:
        original = [dict(sample_post)]
        manager.save_posts(original)
        loaded = manager.load_posts()
        loaded[0]["domain"] = "mutated.com"
        assert original[0]["domain"] == "example.com"

    def test_save_persists_to_disk(self, manager: StorageManager, sample_post: dict) -> None:
        manager.save_posts([sample_post])
        raw = json.loads(manager._posts_path.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        assert len(raw) == 1

    def test_save_valid_json(self, manager: StorageManager, sample_post: dict) -> None:
        manager.save_posts([sample_post])
        # Re-read from disk to confirm valid JSON
        data = json.loads(manager._posts_path.read_text(encoding="utf-8"))
        assert data[0]["title"] == "Example Site"


# =========================================================================
# add_post tests
# =========================================================================

class TestAddPost:

    def test_add_assigns_id(self, manager: StorageManager, sample_post: dict) -> None:
        result = manager.add_post(sample_post)
        assert "id" in result
        assert len(result["id"]) == 36  # UUID format

    def test_add_assigns_discovered_at(self, manager: StorageManager, sample_post: dict) -> None:
        result = manager.add_post(sample_post)
        assert "discovered_at" in result
        assert "T" in result["discovered_at"]  # ISO format

    def test_add_preserves_existing_id(self, manager: StorageManager, sample_post: dict) -> None:
        sample_post["id"] = "custom-id-123"
        result = manager.add_post(sample_post)
        assert result["id"] == "custom-id-123"

    def test_add_preserves_existing_discovered_at(self, manager: StorageManager, sample_post: dict) -> None:
        sample_post["discovered_at"] = "2023-06-15T12:00:00Z"
        result = manager.add_post(sample_post)
        assert result["discovered_at"] == "2023-06-15T12:00:00Z"

    def test_add_returns_new_dict(self, manager: StorageManager, sample_post: dict) -> None:
        result = manager.add_post(sample_post)
        result["domain"] = "mutated.com"
        loaded = manager.load_posts()
        assert loaded[0]["domain"] == "example.com"

    def test_add_does_not_mutate_input(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        assert sample_post.get("id") is None  # original wasn't modified

    def test_add_updates_seen_domains(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        assert "example.com" in manager.get_seen_domains()

    def test_add_multiple_posts(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post({**sample_post, "domain": "a.com"})
        manager.add_post({**sample_post, "domain": "b.com"})
        manager.add_post({**sample_post, "domain": "c.com"})
        assert len(manager.load_posts()) == 3

    def test_add_duplicate_domain_raises(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        with pytest.raises(ValueError, match="Domain already exists"):
            manager.add_post({**sample_post, "domain": "example.com", "title": "Dupe"})

    def test_add_empty_domain_succeeds(self, manager: StorageManager, sample_post: dict) -> None:
        sample_post["domain"] = ""
        result = manager.add_post(sample_post)
        assert result["domain"] == ""

    def test_add_multiple_categories(self, manager: StorageManager, sample_post: dict) -> None:
        for i, cat in enumerate(VALID_CATEGORIES):
            manager.add_post({**sample_post, "domain": f"site{i}.com", "category": cat})
        assert len(manager.load_posts()) == len(VALID_CATEGORIES)


# =========================================================================
# get_post tests
# =========================================================================

class TestGetPost:

    def test_get_existing(self, manager: StorageManager, sample_post: dict) -> None:
        added = manager.add_post(sample_post)
        found = manager.get_post(added["id"])
        assert found is not None
        assert found["domain"] == "example.com"

    def test_get_returns_new_dict(self, manager: StorageManager, sample_post: dict) -> None:
        added = manager.add_post(sample_post)
        found = manager.get_post(added["id"])
        found["domain"] = "mutated.com"
        found_again = manager.get_post(added["id"])
        assert found_again["domain"] == "example.com"

    def test_get_nonexistent(self, manager: StorageManager) -> None:
        assert manager.get_post("no-such-id") is None

    def test_get_empty_id(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        assert manager.get_post("") is None

    def test_get_after_save(self, manager: StorageManager, sample_post: dict) -> None:
        added = manager.add_post(sample_post)
        manager.save_posts(manager.load_posts())
        found = manager.get_post(added["id"])
        assert found is not None


# =========================================================================
# get_latest_posts tests
# =========================================================================

class TestGetLatestPosts:

    def test_empty(self, manager: StorageManager) -> None:
        assert manager.get_latest_posts() == []

    def test_all_posts_within_limit(self, manager_with_posts: StorageManager) -> None:
        latest = manager_with_posts.get_latest_posts(limit=10)
        assert len(latest) == 5

    def test_limit_works(self, manager_with_posts: StorageManager) -> None:
        latest = manager_with_posts.get_latest_posts(limit=2)
        assert len(latest) == 2

    def test_sorted_descending(self, manager_with_posts: StorageManager) -> None:
        latest = manager_with_posts.get_latest_posts(limit=5)
        dates = [p["discovered_at"] for p in latest]
        assert dates == sorted(dates, reverse=True)

    def test_default_limit(self, manager_with_posts: StorageManager) -> None:
        # Default is 10, we only have 5
        latest = manager_with_posts.get_latest_posts()
        assert len(latest) == 5

    def test_returns_new_dicts(self, manager_with_posts: StorageManager) -> None:
        latest = manager_with_posts.get_latest_posts(limit=1)
        latest[0]["domain"] = "mutated.com"
        again = manager_with_posts.get_latest_posts(limit=1)
        assert again[0]["domain"] != "mutated.com"


# =========================================================================
# get_posts_by_category tests
# =========================================================================

class TestGetPostsByCategory:

    def test_matching_category(self, manager_with_posts: StorageManager) -> None:
        interactive = manager_with_posts.get_posts_by_category("Interactive")
        assert len(interactive) == 2
        assert all(p["category"] == "Interactive" for p in interactive)

    def test_no_matching_category(self, manager_with_posts: StorageManager) -> None:
        games = manager_with_posts.get_posts_by_category("Games")
        assert games == []

    def test_returns_new_dicts(self, manager_with_posts: StorageManager) -> None:
        result = manager_with_posts.get_posts_by_category("Interactive")
        result[0]["category"] = "mutated"
        again = manager_with_posts.get_posts_by_category("Interactive")
        assert all(p["category"] == "Interactive" for p in again)

    def test_single_match(self, manager_with_posts: StorageManager) -> None:
        tools = manager_with_posts.get_posts_by_category("Tools")
        assert len(tools) == 1
        assert tools[0]["title"] == "Site 3"

    def test_case_sensitive(self, manager_with_posts: StorageManager) -> None:
        result = manager_with_posts.get_posts_by_category("interactive")
        assert result == []


# =========================================================================
# domain_exists tests
# =========================================================================

class TestDomainExists:

    def test_exists_in_seen(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        assert manager.domain_exists("example.com")

    def test_not_exists(self, manager: StorageManager) -> None:
        assert not manager.domain_exists("nonexistent.com")

    def test_exists_in_posts_only(self, tmp_path: Path, sample_post: dict) -> None:
        """Domain in posts.json but not in seen_domains.json."""
        posts_file = tmp_path / "posts.json"
        posts_file.write_text(
            json.dumps([{**sample_post, "domain": "orphan.com"}]),
            encoding="utf-8",
        )
        m = StorageManager(posts_path=posts_file, seen_domains_path=tmp_path / "seen.json")
        assert m.domain_exists("orphan.com")

    def test_empty_domain(self, manager: StorageManager) -> None:
        assert not manager.domain_exists("")

    def test_domain_case_sensitive(self, manager: StorageManager, sample_post: dict) -> None:
        manager.add_post(sample_post)
        # Domain stored as-is, check is exact match
        assert manager.domain_exists("example.com")
        assert not manager.domain_exists("Example.com")


# =========================================================================
# cleanup_old_posts tests
# =========================================================================

class TestCleanupOldPosts:

    def test_no_cleanup_when_under_limit(self, manager_with_posts: StorageManager) -> None:
        archived = manager_with_posts.cleanup_old_posts(keep_count=10)
        assert archived == []
        assert len(manager_with_posts.load_posts()) == 5

    def test_cleanup_removes_oldest(self, manager_with_posts: StorageManager) -> None:
        archived = manager_with_posts.cleanup_old_posts(keep_count=3)
        assert len(archived) == 2
        remaining = manager_with_posts.load_posts()
        assert len(remaining) == 3
        # Oldest should be archived (site1.com = 2024-01-01)
        archived_domains = {p["domain"] for p in archived}
        assert "site1.com" in archived_domains
        assert "site2.com" in archived_domains

    def test_cleanup_keeps_newest(self, manager_with_posts: StorageManager) -> None:
        manager_with_posts.cleanup_old_posts(keep_count=3)
        remaining = manager_with_posts.load_posts()
        remaining_domains = {p["domain"] for p in remaining}
        assert "site3.com" in remaining_domains
        assert "site4.com" in remaining_domains
        assert "site5.com" in remaining_domains

    def test_cleanup_returns_new_dicts(self, manager_with_posts: StorageManager) -> None:
        archived = manager_with_posts.cleanup_old_posts(keep_count=3)
        archived[0]["domain"] = "mutated.com"
        # Re-cleanup should still have original archived
        archived2 = manager_with_posts.cleanup_old_posts(keep_count=3)
        assert len(archived2) == 0  # nothing more to archive

    def test_cleanup_default_keep_count(self, tmp_path: Path, sample_post: dict) -> None:
        m = StorageManager(
            posts_path=tmp_path / "posts.json",
            seen_domains_path=tmp_path / "seen.json",
            default_keep_count=3,
        )
        for i in range(5):
            m.add_post({**sample_post, "domain": f"site{i}.com"})
        archived = m.cleanup_old_posts()
        assert len(archived) == 2
        assert len(m.load_posts()) == 3

    def test_cleanup_exact_limit(self, manager_with_posts: StorageManager) -> None:
        """Exactly at the limit - nothing removed."""
        archived = manager_with_posts.cleanup_old_posts(keep_count=5)
        assert archived == []
        assert len(manager_with_posts.load_posts()) == 5

    def test_cleanup_one_post(self, tmp_path: Path, sample_post: dict) -> None:
        m = StorageManager(
            posts_path=tmp_path / "posts.json",
            seen_domains_path=tmp_path / "seen.json",
        )
        m.add_post(sample_post)
        archived = m.cleanup_old_posts(keep_count=1)
        assert archived == []
        assert len(m.load_posts()) == 1

    def test_cleanup_zero_keep(self, manager_with_posts: StorageManager) -> None:
        archived = manager_with_posts.cleanup_old_posts(keep_count=0)
        assert len(archived) == 5
        assert len(manager_with_posts.load_posts()) == 0


# =========================================================================
# seen domains persistence tests
# =========================================================================

class TestSeenDomainsPersistence:

    def test_seen_domains_persist_across_instances(self, tmp_path: Path, sample_post: dict) -> None:
        posts_file = tmp_path / "posts.json"
        seen_file = tmp_path / "seen.json"

        m1 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        m1.add_post(sample_post)
        assert "example.com" in m1.get_seen_domains()

        # New instance loads from disk
        m2 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        assert "example.com" in m2.get_seen_domains()

    def test_seen_domains_from_file(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text(json.dumps(["alpha.com", "beta.org"]), encoding="utf-8")
        m = StorageManager(posts_path=tmp_path / "posts.json", seen_domains_path=seen_file)
        assert m.get_seen_domains() == {"alpha.com", "beta.org"}

    def test_missing_seen_file(self, tmp_path: Path) -> None:
        m = StorageManager(
            posts_path=tmp_path / "posts.json",
            seen_domains_path=tmp_path / "nonexistent.json",
        )
        assert m.get_seen_domains() == set()

    def test_corrupt_seen_file(self, tmp_path: Path) -> None:
        seen_file = tmp_path / "seen.json"
        seen_file.write_text("NOT JSON!!!", encoding="utf-8")
        m = StorageManager(posts_path=tmp_path / "posts.json", seen_domains_path=seen_file)
        assert m.get_seen_domains() == set()


# =========================================================================
# Integration / roundtrip tests
# =========================================================================

class TestRoundtrip:

    def test_add_load_get_cycle(self, tmp_path: Path, sample_post: dict) -> None:
        """Full lifecycle: create, add, reload, query."""
        posts_file = tmp_path / "posts.json"
        seen_file = tmp_path / "seen.json"

        # First instance adds posts
        m1 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        added = m1.add_post(sample_post)
        m1.add_post({**sample_post, "domain": "other.com", "title": "Other"})

        # Second instance loads them
        m2 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        assert len(m2.load_posts()) == 2
        found = m2.get_post(added["id"])
        assert found is not None
        assert found["domain"] == "example.com"
        assert m2.domain_exists("other.com")

    def test_multiple_save_load_cycles(self, manager: StorageManager, sample_post: dict) -> None:
        for i in range(5):
            manager.add_post({**sample_post, "domain": f"cycle{i}.com"})
            loaded = manager.load_posts()
            assert len(loaded) == i + 1

    def test_cleanup_then_add(self, manager: StorageManager, sample_post: dict) -> None:
        """After cleanup, can still add new posts."""
        for i in range(5):
            manager.add_post({**sample_post, "domain": f"pre{i}.com"})
        manager.cleanup_old_posts(keep_count=2)
        manager.add_post({**sample_post, "domain": "post-cleanup.com"})
        loaded = manager.load_posts()
        assert len(loaded) == 3
        assert manager.domain_exists("post-cleanup.com")


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:

    def test_post_with_all_fields(self, manager: StorageManager) -> None:
        full_post = {
            "id": "test-uuid-123",
            "url": "https://example.com/page",
            "canonical_url": "https://example.com/page",
            "domain": "example.com",
            "title": "Full Post",
            "description": "Complete post with all fields",
            "category": "Art",
            "source": "github",
            "source_type": "creative_coding",
            "score": 95,
            "screenshot_path": "screenshots/example-com.webp",
            "ai_description": "Amazing generative art!",
            "why_interesting": "Pushes boundaries of web art",
            "discovered_at": "2024-06-15T10:30:00Z",
            "published_at": "2024-06-15T12:00:00Z",
        }
        result = manager.add_post(full_post)
        assert result["id"] == "test-uuid-123"
        assert result["published_at"] == "2024-06-15T12:00:00Z"

    def test_post_with_extra_fields(self, manager: StorageManager, sample_post: dict) -> None:
        """Unknown fields are preserved."""
        sample_post["custom_field"] = "custom_value"
        result = manager.add_post(sample_post)
        assert result["custom_field"] == "custom_value"

    def test_unicode_in_post(self, manager: StorageManager, sample_post: dict) -> None:
        sample_post["title"] = "日本語サイト 🎨"
        sample_post["description"] = "Ünïcödé test"
        result = manager.add_post(sample_post)
        loaded = manager.get_post(result["id"])
        assert loaded["title"] == "日本語サイト 🎨"

    def test_large_post_list(self, manager: StorageManager, sample_post: dict) -> None:
        """Stress test with many posts."""
        for i in range(100):
            manager.add_post({**sample_post, "domain": f"stress{i}.com"})
        assert len(manager.load_posts()) == 100

    def test_cleanup_then_reload(self, tmp_path: Path, sample_post: dict) -> None:
        """Cleanup persists correctly and reloads properly."""
        posts_file = tmp_path / "posts.json"
        seen_file = tmp_path / "seen.json"

        m1 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        for i in range(10):
            m1.add_post({**sample_post, "domain": f"reload{i}.com"})
        m1.cleanup_old_posts(keep_count=3)

        m2 = StorageManager(posts_path=posts_file, seen_domains_path=seen_file)
        assert len(m2.load_posts()) == 3

    def test_valid_categories(self) -> None:
        """Ensure VALID_CATEGORIES matches spec."""
        expected = {"Interactive", "Weird", "Tools", "Games", "Art", "Educational"}
        assert VALID_CATEGORIES == expected
