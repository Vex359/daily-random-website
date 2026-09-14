"""Tests for screenshot capture.

Verifies that Playwright-based screenshots are captured
and saved correctly for featured websites.  All browser
interactions are mocked — no real navigation occurs.
"""

from __future__ import annotations

import random as _random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.screenshot.capture import (
    ScreenshotCapture,
    _create_placeholder,
    url_to_filename,
    MAX_WIDTH_PX,
    DEFAULT_QUALITY,
    PLACEHOLDER_SIZE,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _fake_screenshot_write(path: str, **kwargs: object) -> None:
    """Side-effect that writes a real WebP image to *path*."""
    img = Image.new("RGB", (1280, 800), (100, 150, 200))
    img.save(str(path), format="WEBP", quality=80)


def _make_mock_page() -> MagicMock:
    """Create a mock Playwright page that produces a valid WebP screenshot."""
    page = MagicMock()
    page.screenshot.side_effect = _fake_screenshot_write
    return page


def _make_mock_browser(page: MagicMock | None = None) -> MagicMock:
    """Create a mock Playwright browser returning *page* from ``new_page``."""
    browser = MagicMock()
    browser.new_page.return_value = page or _make_mock_page()
    return browser


def _patch_pw(mock_pw_cls: MagicMock, browser: MagicMock) -> MagicMock:
    """Wire ``sync_playwright`` mock so ``_ensure_browser`` works.

    Returns the mock page object from ``browser.new_page()`` for assertions.

    ``sync_playwright()`` returns ``mock_pw_cls.return_value``,
    whose ``.start()`` returns the same object, and whose
    ``.chromium.launch()`` returns *browser*.
    """
    pw_instance = mock_pw_cls.return_value  # what sync_playwright() returns
    pw_instance.start.return_value = pw_instance
    pw_instance.chromium.launch.return_value = browser
    return browser.new_page.return_value


# ======================================================================
# url_to_filename helper tests
# ======================================================================


class TestUrlToFilename:
    """Tests for the URL → filename converter."""

    def test_basic_url(self):
        assert url_to_filename("https://www.example.com/page") == "example-com.webp"

    def test_bare_domain(self):
        assert url_to_filename("https://example.com") == "example-com.webp"

    def test_subdomain(self):
        assert url_to_filename("https://blog.example.com") == "blog-example-com.webp"

    def test_with_port(self):
        assert url_to_filename("https://example.com:8080/path") == "example-com.webp"

    def test_no_scheme(self):
        result = url_to_filename("example.com")
        assert result.endswith(".webp")
        assert "example" in result

    def test_complex_path(self):
        result = url_to_filename("https://news.ycombinator.com/item?id=123")
        assert result == "news-ycombinator-com.webp"

    def test_empty_string(self):
        result = url_to_filename("")
        assert result.endswith(".webp")

    def test_www_stripped(self):
        """Only the leading www. prefix is stripped."""
        assert url_to_filename("https://www.www.example.com") == "www-example-com.webp"


# ======================================================================
# _create_placeholder helper tests
# ======================================================================


class TestCreatePlaceholder:
    """Tests for the placeholder image generator."""

    def test_creates_webp_file(self, tmp_path: Path):
        out = tmp_path / "placeholder.webp"
        result = _create_placeholder(out)
        assert result == out
        assert out.exists()
        img = Image.open(out)
        assert img.format == "WEBP"

    def test_placeholder_dimensions(self, tmp_path: Path):
        out = tmp_path / "placeholder.webp"
        _create_placeholder(out)
        img = Image.open(out)
        assert img.size == PLACEHOLDER_SIZE

    def test_placeholder_color(self, tmp_path: Path):
        out = tmp_path / "placeholder.webp"
        _create_placeholder(out)
        img = Image.open(out)
        pixel = img.getpixel((0, 0))
        assert pixel[:3] == (240, 240, 240)


# ======================================================================
# ScreenshotCapture init / lifecycle tests
# ======================================================================


class TestScreenshotCaptureInit:
    """Test constructor and lifecycle."""

    def test_default_headless(self):
        cap = ScreenshotCapture()
        assert cap._headless is True
        assert cap._browser is None

    def test_custom_headless(self):
        cap = ScreenshotCapture(headless=False)
        assert cap._headless is False

    @patch("src.screenshot.capture.sync_playwright")
    def test_context_manager(self, mock_pw_cls: MagicMock):
        browser = _make_mock_browser()
        _patch_pw(mock_pw_cls, browser)

        with ScreenshotCapture() as cap:
            assert cap._browser is not None

        browser.close.assert_called_once()

    def test_close_idempotent(self):
        cap = ScreenshotCapture.__new__(ScreenshotCapture)
        cap._headless = True
        cap._pw = None
        cap._browser = None
        cap.close()  # should not raise

    def test_close_with_browser(self):
        cap = ScreenshotCapture.__new__(ScreenshotCapture)
        cap._headless = True
        cap._pw = MagicMock()
        cap._browser = MagicMock()
        cap.close()
        assert cap._browser is None
        assert cap._pw is None


# ======================================================================
# capture_screenshot tests (mocked Playwright)
# ======================================================================


class TestCaptureScreenshot:
    """Test capture_screenshot with mocked Playwright."""

    @patch("src.screenshot.capture.sync_playwright")
    def test_captures_to_webp(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        result = cap.capture_screenshot("https://example.com", out)
        cap.close()

        assert result == out
        assert out.exists()
        img = Image.open(out)
        assert img.format == "WEBP"

    @patch("src.screenshot.capture.sync_playwright")
    def test_navigates_with_networkidle(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out)
        cap.close()

        page.goto.assert_called_once()
        _, kwargs = page.goto.call_args
        assert kwargs.get("wait_until") == "networkidle"

    @patch("src.screenshot.capture.sync_playwright")
    def test_full_page_false(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out)
        cap.close()

        _, kwargs = page.screenshot.call_args
        assert kwargs.get("full_page") is False

    @patch("src.screenshot.capture.sync_playwright")
    def test_webp_quality(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out)
        cap.close()

        _, kwargs = page.screenshot.call_args
        assert kwargs.get("type") == "webp"
        assert kwargs.get("quality") == DEFAULT_QUALITY

    @patch("src.screenshot.capture.sync_playwright")
    def test_viewport_dimensions(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out)
        cap.close()

        _, kwargs = browser.new_page.call_args
        vp = kwargs.get("viewport", {})
        assert vp.get("width") == 1280
        assert vp.get("height") == 800

    @patch("src.screenshot.capture.sync_playwright")
    def test_timeout_propagated(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out, timeout_ms=15000)
        cap.close()

        _, kwargs = page.goto.call_args
        assert kwargs.get("timeout") == 15000

    @patch("src.screenshot.capture.sync_playwright")
    def test_creates_parent_dirs(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "sub" / "dir" / "test.webp"
        browser = _make_mock_browser(_make_mock_page())
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://example.com", out)
        cap.close()

        assert out.exists()


# ======================================================================
# Error handling tests
# ======================================================================


class TestCaptureErrorHandling:
    """Verify graceful fallback when Playwright fails."""

    @patch("src.screenshot.capture.sync_playwright")
    def test_timeout_creates_placeholder(self, mock_pw_cls: MagicMock, tmp_path: Path):
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        out = tmp_path / "timeout.webp"
        page = MagicMock()
        page.goto.side_effect = PlaywrightTimeout("timeout")
        browser = _make_mock_browser(page)
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        result = cap.capture_screenshot("https://slow.example.com", out)
        cap.close()

        assert result == out
        assert out.exists()
        img = Image.open(out)
        assert img.size == PLACEHOLDER_SIZE

    @patch("src.screenshot.capture.sync_playwright")
    def test_playwright_error_creates_placeholder(self, mock_pw_cls: MagicMock, tmp_path: Path):
        from playwright.sync_api import Error as PlaywrightError

        out = tmp_path / "error.webp"
        page = MagicMock()
        page.goto.side_effect = PlaywrightError("net::ERR_NAME_NOT_RESOLVED")
        browser = _make_mock_browser(page)
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        result = cap.capture_screenshot("https://nonexistent.invalid", out)
        cap.close()

        assert out.exists()
        img = Image.open(out)
        assert img.size == PLACEHOLDER_SIZE

    @patch("src.screenshot.capture.sync_playwright")
    def test_page_closed_on_failure(self, mock_pw_cls: MagicMock, tmp_path: Path):
        from playwright.sync_api import Error as PlaywrightError

        out = tmp_path / "err.webp"
        page = MagicMock()
        page.goto.side_effect = PlaywrightError("fail")
        browser = _make_mock_browser(page)
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://bad.example.com", out)
        cap.close()

        page.close.assert_called_once()


# ======================================================================
# Resize and compression tests
# ======================================================================


class TestResizeAndCompress:
    """Verify resize and compress-if-needed logic."""

    @patch("src.screenshot.capture.sync_playwright")
    def test_wide_image_resized(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "wide.webp"
        page = MagicMock()

        def fake_wide_screenshot(path: str, **kwargs: object) -> None:
            img = Image.new("RGB", (1920, 1080), (50, 100, 150))
            img.save(str(path), format="WEBP", quality=95)

        page.screenshot.side_effect = fake_wide_screenshot
        browser = _make_mock_browser(page)
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://wide.example.com", out)
        cap.close()

        img = Image.open(out)
        assert img.width <= MAX_WIDTH_PX

    def test_optimize_image_reduces_size(self, tmp_path: Path):
        inp = tmp_path / "big.png"
        img = Image.new("RGB", (1600, 900), (100, 200, 50))
        img.save(str(inp), format="PNG")

        out = tmp_path / "optimised.webp"
        result = ScreenshotCapture.optimize_image(inp, out, max_size_kb=50)

        assert result == out
        assert out.exists()
        saved = Image.open(out)
        assert saved.width <= MAX_WIDTH_PX
        assert out.stat().st_size <= 50 * 1024

    def test_optimize_image_preserves_small_files(self, tmp_path: Path):
        inp = tmp_path / "tiny.png"
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        img.save(str(inp), format="PNG")

        out = tmp_path / "out.webp"
        ScreenshotCapture.optimize_image(inp, out, max_size_kb=200)
        assert out.exists()

    def test_optimize_image_creates_parent_dirs(self, tmp_path: Path):
        inp = tmp_path / "src.png"
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        img.save(str(inp), format="PNG")

        out = tmp_path / "a" / "b" / "c" / "out.webp"
        ScreenshotCapture.optimize_image(inp, out)
        assert out.exists()


# ======================================================================
# Batch capture tests
# ======================================================================


class TestCaptureBatch:
    """Test capture_batch with mocked Playwright."""

    @patch("src.screenshot.capture.sync_playwright")
    def test_batch_captures_all_urls(self, mock_pw_cls: MagicMock, tmp_path: Path):
        urls = ["https://a.example.com", "https://b.example.com"]
        browser = _make_mock_browser(_make_mock_page())
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        results = cap.capture_batch(urls, tmp_path)
        cap.close()

        assert len(results) == 2
        for r in results:
            assert r.exists()
            assert r.suffix == ".webp"

    @patch("src.screenshot.capture.sync_playwright")
    def test_batch_filenames_match_domains(self, mock_pw_cls: MagicMock, tmp_path: Path):
        urls = ["https://www.example.com", "https://httpbin.org/html"]
        browser = _make_mock_browser(_make_mock_page())
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        results = cap.capture_batch(urls, tmp_path)
        cap.close()

        names = {r.name for r in results}
        assert "example-com.webp" in names
        assert "httpbin-org.webp" in names

    @patch("src.screenshot.capture.sync_playwright")
    def test_batch_empty_list(self, mock_pw_cls: MagicMock, tmp_path: Path):
        cap = ScreenshotCapture()
        results = cap.capture_batch([], tmp_path)
        cap.close()
        assert results == []

    @patch("src.screenshot.capture.sync_playwright")
    def test_batch_creates_output_dir(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out_dir = tmp_path / "new_dir"
        browser = _make_mock_browser(_make_mock_page())
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_batch(["https://example.com"], out_dir)
        cap.close()

        assert out_dir.exists()
        assert list(out_dir.glob("*.webp"))


# ======================================================================
# Integration-style: capture + resize + compress end-to-end
# ======================================================================


class TestEndToEndCapture:
    """Full flow: navigate -> screenshot -> resize -> compress."""

    @patch("src.screenshot.capture.sync_playwright")
    def test_output_under_200kb(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "e2e.webp"
        page = MagicMock()

        def fake_noisy_screenshot(path: str, **kwargs: object) -> None:
            _random.seed(42)
            pixels = [_random.randint(0, 255) for _ in range(1280 * 800 * 3)]
            img = Image.frombytes("RGB", (1280, 800), bytes(pixels))
            img.save(str(path), format="WEBP", quality=95)

        page.screenshot.side_effect = fake_noisy_screenshot
        browser = _make_mock_browser(page)
        _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        result = cap.capture_screenshot("https://noisy.example.com", out)
        cap.close()

        size_kb = result.stat().st_size / 1024
        assert size_kb < 200, f"Screenshot too large: {size_kb:.1f} KB"

    @patch("src.screenshot.capture.sync_playwright")
    def test_auto_scroll_called(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "scroll.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://scroll.example.com", out)
        cap.close()

        page.evaluate.assert_called_once()

    @patch("src.screenshot.capture.sync_playwright")
    def test_wait_after_load(self, mock_pw_cls: MagicMock, tmp_path: Path):
        out = tmp_path / "wait.webp"
        browser = _make_mock_browser(_make_mock_page())
        page = _patch_pw(mock_pw_cls, browser)

        cap = ScreenshotCapture()
        cap.capture_screenshot("https://wait.example.com", out)
        cap.close()

        page.wait_for_timeout.assert_called_once_with(1000)
