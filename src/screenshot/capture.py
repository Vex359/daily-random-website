"""Screenshot capture module.

Takes screenshots of featured websites using Playwright
for visual previews on the daily website.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from playwright.sync_api import Error as PlaywrightError
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
SCREENSHOT_TIMEOUT_MS = 30_000  # 30 seconds
WAIT_AFTER_LOAD_S = 1.0  # extra settle time for fonts / lazy images
SCROLL_STEP_PX = 400
SCROLL_PAUSE_MS = 150
MAX_WIDTH_PX = 1200
DEFAULT_QUALITY = 80
DEFAULT_MAX_SIZE_KB = 200

PLACEHOLDER_COLOR = (240, 240, 240)  # light grey
PLACEHOLDER_SIZE = (MAX_WIDTH_PX, 675)  # 16:10-ish


# ── Helpers ──────────────────────────────────────────────────────────────

def url_to_filename(url: str) -> str:
    """Derive a filesystem-safe filename from a URL.

    Uses the domain name with dots replaced by hyphens, e.g.
    ``https://www.example.com/page`` → ``example-com.webp``.
    """
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    # strip port, www prefix, trailing slash
    domain = domain.split(":")[0]
    if domain.startswith("www."):
        domain = domain[4:]
    # replace non-alphanumeric with hyphens, collapse repeats
    safe = ""
    for ch in domain:
        if ch.isalnum():
            safe += ch
        elif safe and safe[-1] != "-":
            safe += "-"
    safe = safe.strip("-")
    return f"{safe}.webp"


def _create_placeholder(output_path: Path) -> Path:
    """Write a small placeholder image and return its path."""
    img = Image.new("RGB", PLACEHOLDER_SIZE, PLACEHOLDER_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="WEBP", quality=DEFAULT_QUALITY)
    return output_path


# ── Main class ───────────────────────────────────────────────────────────

class ScreenshotCapture:
    """Captures and optimises website screenshots via Playwright.

    Usage::

        cap = ScreenshotCapture()
        path = cap.capture_screenshot("https://example.com", "out/example-com.webp")
        cap.close()

    Or as a context manager::

        with ScreenshotCapture() as cap:
            path = cap.capture_screenshot(url, out)
    """

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._pw = None
        self._browser = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _ensure_browser(self) -> None:
        """Lazily start Playwright + Chromium on first use."""
        if self._browser is not None:
            return
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)

    def close(self) -> None:
        """Shut down the browser and Playwright instance."""
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None

    def __enter__(self) -> ScreenshotCapture:
        self._ensure_browser()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── Core capture ─────────────────────────────────────────────────────

    def capture_screenshot(
        self,
        url: str,
        output_path: str | Path,
        *,
        timeout_ms: int = SCREENSHOT_TIMEOUT_MS,
    ) -> Path:
        """Capture a single viewport screenshot.

        Parameters
        ----------
        url:
            The page URL to visit.
        output_path:
            Where to write the resulting WebP image. Parent directories
            are created automatically.
        timeout_ms:
            Maximum wait for page load in milliseconds (default 30 000).

        Returns
        -------
        Path to the saved screenshot (WebP).
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_browser()
        assert self._browser is not None

        page = self._browser.new_page(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )
        try:
            self._navigate_and_capture(page, url, output, timeout_ms)
        except (PlaywrightTimeout, PlaywrightError, OSError) as exc:
            logger.warning("Screenshot failed for %s: %s — writing placeholder", url, exc)
            _create_placeholder(output)
        finally:
            page.close()

        return output

    def _navigate_and_capture(
        self,
        page: object,
        url: str,
        output: Path,
        timeout_ms: int,
    ) -> None:
        """Navigate, scroll to trigger lazy loads, then screenshot."""
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)  # type: ignore[union-attr]

        # Auto-scroll to trigger lazy-loaded content
        self._auto_scroll(page)  # type: ignore[arg-type]

        # Let fonts and remaining images settle
        page.wait_for_timeout(int(WAIT_AFTER_LOAD_S * 1000))  # type: ignore[union-attr]

        # Capture viewport (NOT full page)
        page.screenshot(  # type: ignore[union-attr]
            path=str(output),
            type="webp",
            quality=DEFAULT_QUALITY,
            full_page=False,
        )

        # Resize if wider than MAX_WIDTH_PX
        self._resize_if_needed(output)

        # Compress if still over budget
        self._compress_if_needed(output, DEFAULT_MAX_SIZE_KB)

    def _auto_scroll(self, page: object) -> None:
        """Scroll the page in increments to trigger lazy loading."""
        page.evaluate(  # type: ignore[union-attr]
            """
            async () => {
                const step = arguments[0];
                const pause = arguments[1];
                const delay = ms => new Promise(r => setTimeout(r, ms));
                let pos = 0;
                const maxScroll = document.body.scrollHeight;
                while (pos < maxScroll) {
                    pos += step;
                    window.scrollTo(0, pos);
                    await delay(pause);
                }
                // Scroll back to top for a clean capture
                window.scrollTo(0, 0);
            }
            """,
            SCROLL_STEP_PX,
            SCROLL_PAUSE_MS,
        )

    # ── Resize / optimise ────────────────────────────────────────────────

    def _resize_if_needed(self, path: Path) -> None:
        """Downscale to MAX_WIDTH_PX if the image is wider."""
        try:
            img = Image.open(path)
            if img.width > MAX_WIDTH_PX:
                ratio = MAX_WIDTH_PX / img.width
                new_h = int(img.height * ratio)
                img = img.resize((MAX_WIDTH_PX, new_h), Image.LANCZOS)
                img.save(str(path), format="WEBP", quality=DEFAULT_QUALITY)
                logger.debug("Resized %s → %dx%d", path.name, MAX_WIDTH_PX, new_h)
        except Exception as exc:
            logger.warning("Resize failed for %s: %s", path.name, exc)

    def _compress_if_needed(self, path: Path, max_kb: int) -> None:
        """Iteratively lower quality to meet the file-size budget."""
        max_bytes = max_kb * 1024
        try:
            if path.stat().st_size <= max_bytes:
                return

            img = Image.open(path)
            for quality in range(DEFAULT_QUALITY - 5, 9, -5):
                img.save(str(path), format="WEBP", quality=quality)
                if path.stat().st_size <= max_bytes:
                    logger.debug(
                        "Compressed %s to %d KB (q=%d)",
                        path.name,
                        path.stat().st_size // 1024,
                        quality,
                    )
                    return
            logger.warning(
                "Could not compress %s under %d KB (best: %d KB)",
                path.name,
                max_kb,
                path.stat().st_size // 1024,
            )
        except Exception as exc:
            logger.warning("Compression failed for %s: %s", path.name, exc)

    # ── Batch capture ────────────────────────────────────────────────────

    def capture_batch(
        self,
        urls: list[str],
        output_dir: str | Path,
        *,
        timeout_ms: int = SCREENSHOT_TIMEOUT_MS,
    ) -> list[Path]:
        """Capture screenshots for multiple URLs.

        Parameters
        ----------
        urls:
            List of page URLs.
        output_dir:
            Directory to write screenshots into.
        timeout_ms:
            Per-URL timeout in milliseconds.

        Returns
        -------
        List of Paths for screenshots that were written (one per URL).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results: list[Path] = []
        for url in urls:
            filename = url_to_filename(url)
            dest = out / filename
            logger.info("Capturing %s → %s", url, dest)
            saved = self.capture_screenshot(url, dest, timeout_ms=timeout_ms)
            results.append(saved)
        return results

    # ── Standalone optimise (public) ─────────────────────────────────────

    @staticmethod
    def optimize_image(
        input_path: str | Path,
        output_path: str | Path,
        max_size_kb: int = DEFAULT_MAX_SIZE_KB,
    ) -> Path:
        """Resize and compress an existing image to WebP.

        Parameters
        ----------
        input_path:
            Source image (any format Pillow can read).
        output_path:
            Destination path (WebP recommended).
        max_size_kb:
            Target maximum file size in KB.

        Returns
        -------
        Path to the optimised image.
        """
        inp = Path(input_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(inp)

        # Downscale if wider than limit
        if img.width > MAX_WIDTH_PX:
            ratio = MAX_WIDTH_PX / img.width
            img = img.resize(
                (MAX_WIDTH_PX, int(img.height * ratio)),
                Image.LANCZOS,
            )

        # Iteratively reduce quality to hit budget
        max_bytes = max_size_kb * 1024
        for quality in range(DEFAULT_QUALITY, 9, -5):
            img.save(str(out), format="WEBP", quality=quality)
            if out.stat().st_size <= max_bytes:
                return out

        # Last resort — save at lowest acceptable quality
        img.save(str(out), format="WEBP", quality=10)
        return out


# ── CLI entry point ──────────────────────────────────────────────────────

def _main() -> None:
    """Capture test screenshots when run as ``python -m src.screenshot.capture``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://www.wikipedia.org",
    ]

    screenshots_dir = Path(__file__).resolve().parent.parent.parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    with ScreenshotCapture() as cap:
        results = cap.capture_batch(test_urls, screenshots_dir)

    for url, path in zip(test_urls, results):
        size_kb = path.stat().st_size / 1024
        print(f"  ✓ {url}  →  {path.name}  ({size_kb:.1f} KB)")

    print(f"\n{len(results)} screenshot(s) saved to {screenshots_dir}")


if __name__ == "__main__":
    _main()
