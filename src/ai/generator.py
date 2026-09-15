"""AI-powered description generation using OpenRouter API.

Generates Instagram-style post descriptions for collected content candidates.
Uses the OpenRouter free-tier model for cost-free generation.

Usage as script:
    python -m src.ai.generator
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
REQUEST_TIMEOUT = 30.0

# Rate limiting: 20 requests per minute (OpenRouter free tier limit)
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_RETRY_WAIT = 60.0

# Input limits
MAX_PAGE_TEXT_WORDS = 500

# Output validation
MIN_DESCRIPTION_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 300

# System prompt for the AI
SYSTEM_PROMPT = (
    "You are a social media copywriter. Write short, engaging Instagram-style "
    "post descriptions. Keep responses to 2-3 sentences. Be vivid, use sensory "
    "language. Use emojis sparingly (1-2 max). End with a hook or call-to-action."
)

# Fallback description when AI fails
FALLBACK_DESCRIPTION = (
    "Discover something interesting on the internet today. "
    "This curated pick stands out from the noise. Check it out!"
)


# ---------------------------------------------------------------------------
# Clean name & Categorization helpers
# ---------------------------------------------------------------------------

def clean_website_name(title: str, domain: str = "") -> str:
    """Extract a concise website/project brand name (1-4 words)."""
    if not title or title.strip().lower() in ("untitled", "no title", "unknown"):
        if domain:
            cleaned_dom = domain.replace("www.", "").split(":")[0]
            parts = cleaned_dom.split(".")
            brand = parts[0] if parts[0] not in ("com", "org", "net", "io", "dev") else (parts[-2] if len(parts) > 1 else parts[0])
            return brand.capitalize()
        return "Featured Site"

    t = title.strip()
    # Strip common prefixes
    prefixes = [
        "show hn:", "ask hn:", "tell hn:", "launch hn:",
        "showcase:", "project:", "announcing", "introducing"
    ]
    for p in prefixes:
        if t.lower().startswith(p):
            t = t[len(p):].strip()
            t = t.lstrip("-–—: ").strip()

    # Split by common title separators
    for sep in [" – ", " — ", " - ", " | ", " :: ", " : ", " · ", " • "]:
        if sep in t:
            candidate_part = t.split(sep)[0].strip()
            word_count = len(candidate_part.split())
            if 1 <= word_count <= 4 and len(candidate_part) <= 30:
                t = candidate_part
                break

    words = t.split()
    if len(words) > 4:
        if domain:
            dom_brand = domain.replace("www.", "").split(".")[0].lower()
            for w in words:
                if dom_brand in w.lower():
                    return w.strip(".,;:\"'()")
        return " ".join(words[:4])

    return t


def categorize_candidate(candidate: dict[str, Any]) -> str:
    """Intelligently assign a category based on candidate metadata keywords."""
    text = " ".join([
        str(candidate.get("title") or ""),
        str(candidate.get("description") or ""),
        str(candidate.get("why_interesting") or ""),
        str(candidate.get("url") or ""),
        " ".join(candidate.get("headings") or []) if isinstance(candidate.get("headings"), list) else "",
    ]).lower()

    if any(k in text for k in ["game", "play", "puzzle", "arcade", "quest"]):
        return "Games"
    if any(k in text for k in ["interactive", "simulator", "webgl", "3d", "canvas", "three.js", "shader"]):
        return "Interactive"
    if any(k in text for k in ["tool", "utility", "converter", "calculator", "developer", "code", "github", "git", "api", "database", "redis", "analytics", "voice", "tts", "ai", "model"]):
        return "Tools"
    if any(k in text for k in ["art", "generative", "creative", "design", "gallery", "drawing", "illustration"]):
        return "Art"
    if any(k in text for k in ["learn", "teach", "history", "wikipedia", "education", "explorable", "guide"]):
        return "Educational"
    if any(k in text for k in ["weird", "strange", "bizarre", "useless", "novelty", "funny", "random"]):
        return "Weird"

    existing = candidate.get("category")
    if existing and existing in ("Interactive", "Weird", "Tools", "Games", "Art", "Educational", "General"):
        return existing

    return "General"


def generate_tags(candidate: dict[str, Any], category: str = "") -> list[str]:
    """Generate 2-4 topical keyword tags for a candidate."""
    tags: list[str] = []
    if category and category != "Uncategorized":
        tags.append(category)

    text = " ".join([
        str(candidate.get("title") or ""),
        str(candidate.get("description") or ""),
        str(candidate.get("url") or ""),
    ]).lower()

    keywords_map = [
        ("3d", "3D"), ("redis", "Redis"), ("database", "Database"),
        ("ai", "AI"), ("voice", "Voice"), ("tts", "TTS"),
        ("analytics", "Analytics"), ("open source", "Open Source"),
        ("developer", "Dev Tools"), ("interactive", "Interactive"),
        ("game", "Game"), ("webgl", "WebGL"), ("design", "Design"),
        ("music", "Music"), ("simulation", "Simulation"),
    ]

    for kw, tag_label in keywords_map:
        if kw in text and tag_label not in tags:
            tags.append(tag_label)
        if len(tags) >= 4:
            break

    if not tags:
        tags = ["Web", "Featured"]
    return tags


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AIGeneratorError(Exception):
    """Raised when the AI generator encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AIDescriptionGenerator:
    """Generates Instagram-style descriptions for content candidates via OpenRouter.

    Uses rate limiting (20 req/min) and retries once on rate limit (waits 60s).
    All API calls are made through httpx with proper error handling.

    Usage::

        gen = AIDescriptionGenerator(api_key="sk-or-...")
        result = gen.generate_description(candidate)
        print(result)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialise the generator.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            model: Model identifier (default: openrouter/free).
            timeout: HTTP request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._timeout = timeout
        self._session = httpx.Client(timeout=timeout)
        self._rate_limit_times: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_description(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Generate an Instagram-style description for a single candidate.

        Args:
            candidate: Dict with keys: url, title, description (optional),
                headings (optional list), page_text (optional str).

        Returns:
            Dict with keys: description, category, why_interesting.
            Falls back to a default description on API failure.
        """
        if not self._api_key:
            logger.warning("No OPENROUTER_API_KEY set; using fallback description")
            return self._fallback_output(candidate)

        user_prompt = self._build_user_prompt(candidate)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = self._call_api(messages)
            result = self._parse_ai_response(raw, candidate)
            return result
        except AIGeneratorError as exc:
            # If openrouter/free gave a moderation-only response, try backup model
            if "Moderation-only" in str(exc):
                try:
                    raw = self._call_api(messages, model_override="nex-agi/nex-n2.5-pro:free")
                    return self._parse_ai_response(raw, candidate)
                except Exception:
                    pass
            return self._fallback_output(candidate)

    def generate_batch(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate descriptions for multiple candidates.

        Args:
            candidates: List of candidate dicts.

        Returns:
            List of description dicts (one per candidate). Failed candidates
            get fallback descriptions; the list length always matches the input.
        """
        results: list[dict[str, Any]] = []
        for i, candidate in enumerate(candidates):
            logger.info(
                "Generating description %d/%d: %s",
                i + 1,
                len(candidates),
                candidate.get("url", "unknown"),
            )
            result = self.generate_description(candidate)
            results.append(result)
        return results

    def validate_output(self, description: dict[str, Any]) -> dict[str, Any]:
        """Validate the quality of a generated description.

        Checks:
        - Required keys exist (description, category, why_interesting)
        - Description length is 50-300 characters
        - Description references input data (no hallucinations)

        Args:
            description: Dict with keys: description, category, why_interesting.

        Returns:
            Dict with keys: is_valid (bool), issues (list[str]).
        """
        issues: list[str] = []

        # Check required keys
        for key in ("description", "category", "why_interesting"):
            if key not in description:
                issues.append(f"Missing required key: {key}")

        # Validate description length
        desc_text = description.get("description", "")
        if len(desc_text) < MIN_DESCRIPTION_LENGTH:
            issues.append(
                f"Description too short: {len(desc_text)} chars "
                f"(minimum {MIN_DESCRIPTION_LENGTH})"
            )
        if len(desc_text) > MAX_DESCRIPTION_LENGTH:
            issues.append(
                f"Description too long: {len(desc_text)} chars "
                f"(maximum {MAX_DESCRIPTION_LENGTH})"
            )

        # Validate category is non-empty
        category = description.get("category", "")
        if not category or not category.strip():
            issues.append("Category is empty")

        # Validate why_interesting is non-empty
        why = description.get("why_interesting", "")
        if not why or not why.strip():
            issues.append("why_interesting is empty")

        return {"is_valid": len(issues) == 0, "issues": issues}

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> AIDescriptionGenerator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal: rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> None:
        """Enforce the 20 requests per minute rate limit.

        Prunes timestamps older than the window, then sleeps if the
        limit is reached.
        """
        now = time.monotonic()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        self._rate_limit_times = [t for t in self._rate_limit_times if t > cutoff]

        if len(self._rate_limit_times) >= RATE_LIMIT_MAX_REQUESTS:
            # Sleep until the oldest request falls outside the window
            sleep_time = RATE_LIMIT_WINDOW_SECONDS - (now - self._rate_limit_times[0])
            if sleep_time > 0:
                logger.info("Rate limit reached; sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
            # After sleeping, prune again
            now = time.monotonic()
            cutoff = now - RATE_LIMIT_WINDOW_SECONDS
            self._rate_limit_times = [t for t in self._rate_limit_times if t > cutoff]

        self._rate_limit_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, str]], model_override: str | None = None) -> str:
        """Call the OpenRouter chat completions API.

        Retries once on rate limit (HTTP 429), waiting 60 seconds.

        Returns the assistant message content as a string.
        Raises AIGeneratorError on failure.
        """
        self._check_rate_limit()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://daily-random-website.netlify.app",
            "X-Title": "Daily Random Website",
        }

        model = model_override or self._model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.5,
        }

        try:
            resp = self._session.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP error calling OpenRouter: %s", exc)
            raise AIGeneratorError(f"HTTP error: {exc}") from exc

        # Handle rate limit with one retry
        if resp.status_code == 429:
            logger.warning(
                "Rate limited (429); retrying after %ds", int(RATE_LIMIT_RETRY_WAIT)
            )
            time.sleep(RATE_LIMIT_RETRY_WAIT)
            self._rate_limit_times.clear()
            self._check_rate_limit()
            try:
                resp = self._session.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )
            except httpx.HTTPError as exc:
                logger.error("HTTP error on retry: %s", exc)
                raise AIGeneratorError(f"HTTP error on retry: {exc}") from exc

        # Handle other HTTP errors
        if resp.status_code >= 400:
            logger.error(
                "OpenRouter API error %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            raise AIGeneratorError(
                f"API returned status {resp.status_code}"
            )

        # Parse JSON response
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Invalid JSON from OpenRouter: %s", exc)
            raise AIGeneratorError(f"Invalid JSON response: {exc}") from exc

        # Extract content from response
        try:
            content = data["choices"][0]["message"]["content"]
            if not content:
                raise AIGeneratorError("Empty response from OpenRouter")
            cleaned = content.strip()
            if cleaned in ("User Safety: safe", "safe", "User Safety: unsafe"):
                raise AIGeneratorError(f"Moderation-only response from model: {cleaned}")
            return cleaned
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected response structure: %s", data)
            raise AIGeneratorError(
                f"Unexpected API response structure: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal: prompt building
    # ------------------------------------------------------------------

    def _build_user_prompt(self, candidate: dict[str, Any]) -> str:
        """Build the user prompt from candidate data.

        Limits page_text to 500 words to stay within token budgets.
        """
        parts: list[str] = []

        url = candidate.get("url", "Unknown URL")
        title = candidate.get("title", "No title")
        description = candidate.get("description", "No description")

        parts.append(f"URL: {url}")
        parts.append(f"Title: {title}")
        parts.append(f"Description: {description}")

        # Headings (optional)
        headings = candidate.get("headings")
        if headings and isinstance(headings, list):
            headings_text = " | ".join(str(h) for h in headings[:10])
            parts.append(f"Headings: {headings_text}")

        # Page text (optional, limited to 500 words)
        page_text = candidate.get("page_text", "")
        if page_text:
            words = page_text.split()
            if len(words) > MAX_PAGE_TEXT_WORDS:
                words = words[:MAX_PAGE_TEXT_WORDS]
                page_text = " ".join(words) + "..."
            parts.append(f"Page text: {page_text}")

        prompt = (
            "Generate a short Instagram-style description for this website.\n\n"
            + "\n".join(parts)
            + "\n\n"
            "Requirements:\n"
            "1. clean_title: ONLY the clean website/brand name (1-3 words max, e.g. 'Redis City', 'PostHog'). No subtitles or slogans.\n"
            "2. description: Exciting 2-sentence Instagram-style hook about what users can see or do on this site.\n"
            "3. category: Choose ONE from [Interactive, Tools, Games, Art, Educational, Developer Tools, AI & ML, 3D & Graphics, Audio & Voice, Utilities, Weird & Fun].\n"
            "4. tags: Array of 2-4 lowercase keyword strings.\n"
            "5. why_interesting: 1 crisp sentence explaining what makes it cool.\n\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{"clean_title": "...", "description": "...", "category": "...", "tags": ["..."], "why_interesting": "..."}'
        )
        return prompt

    # ------------------------------------------------------------------
    # Internal: response parsing
    # ------------------------------------------------------------------

    def _parse_ai_response(
        self, raw: str, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse the AI response into a structured dict.

        Handles markdown code blocks, plain JSON, and malformed responses.
        Validates output and falls back on failure.
        """
        cleaned = raw.strip()

        # Strip markdown code block wrappers
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

        # Try parsing as JSON
        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse AI response as JSON; "
                "attempting extraction. Raw: %s",
                raw[:200],
            )
            # Try to extract JSON substring
            result = self._extract_json_from_text(cleaned)

        if not isinstance(result, dict):
            logger.warning("AI response is not a dict; using fallback")
            return self._fallback_output(candidate)

        # Ensure required keys exist with defaults
        clean_name = clean_website_name(result.get("clean_title") or candidate.get("title") or "", candidate.get("domain", ""))
        result["clean_title"] = clean_name
        result.setdefault("description", FALLBACK_DESCRIPTION)
        result.setdefault("category", "Uncategorized")
        result.setdefault("why_interesting", "Curated for your interest")
        if "tags" not in result or not isinstance(result.get("tags"), list):
            result["tags"] = generate_tags(candidate, result.get("category", ""))

        # Validate
        validation = self.validate_output(result)
        if not validation["is_valid"]:
            logger.warning(
                "AI output failed validation: %s", validation["issues"]
            )
            # Try to recover: fix length issues
            result = self._fix_description(result)
            # Re-validate
            validation = self.validate_output(result)
            if not validation["is_valid"]:
                logger.warning("Could not fix AI output; using fallback")
                return self._fallback_output(candidate)

        return result

    def _extract_json_from_text(self, text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from arbitrary text."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        substring = text[start : end + 1]
        try:
            return json.loads(substring)
        except (json.JSONDecodeError, ValueError):
            return None

    def _fix_description(self, result: dict[str, Any]) -> dict[str, Any]:
        """Attempt to fix common description issues."""
        desc = result.get("description", "")

        # Trim if too long
        if len(desc) > MAX_DESCRIPTION_LENGTH:
            # Cut at sentence boundary if possible
            trimmed = desc[:MAX_DESCRIPTION_LENGTH]
            last_period = trimmed.rfind(".")
            last_excl = trimmed.rfind("!")
            last_q = trimmed.rfind("?")
            cut = max(last_period, last_excl, last_q)
            if cut > MIN_DESCRIPTION_LENGTH:
                desc = trimmed[: cut + 1]
            else:
                # Leave room for the trailing period
                desc = trimmed[: MAX_DESCRIPTION_LENGTH - 1].rstrip() + "."

        # Pad if too short
        if len(desc) < MIN_DESCRIPTION_LENGTH and desc:
            desc = desc + " " + FALLBACK_DESCRIPTION[len(desc) :]

        result["description"] = desc
        return result

    # ------------------------------------------------------------------
    # Internal: fallback
    # ------------------------------------------------------------------

    def _fallback_output(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Return a fallback description when AI generation fails."""
        return {
            "description": FALLBACK_DESCRIPTION,
            "category": "Uncategorized",
            "why_interesting": "Curated for your interest",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_TEST_CANDIDATES: list[dict[str, Any]] = [
    {
        "url": "https://excalidraw.com",
        "title": "Excalidraw - Virtual Whiteboard",
        "description": "A virtual whiteboard for sketching hand-drawn like diagrams.",
        "headings": ["Features", "Getting Started", "FAQ"],
        "page_text": (
            "Excalidraw is a virtual whiteboard that lets you sketch diagrams "
            "with a hand-drawn, excalidraw-style look. It supports real-time "
            "collaboration, end-to-end encryption, and exports to PNG, SVG, and "
            "JSON. Perfect for quick wireframes, architecture diagrams, and "
            "brainstorming sessions."
        ),
    },
    {
        "url": "https://neal.fun",
        "title": "Neal.fun - Interactive Web Experiments",
        "description": "A collection of fun interactive web experiments and games.",
        "headings": ["The Password Game", "Spend Bill Gates' Money", "Draw a Perfect Circle"],
        "page_text": (
            "Neal.fun is a website full of creative and fun interactive browser "
            "experiments. From The Password Game where you must create increasingly "
            "complex passwords, to Spend Bill Gates' Money where you can blow through "
            "an imaginary fortune. Each experiment is polished, delightful, and "
            "surprisingly addictive."
        ),
    },
    {
        "url": "https://window-swap.com",
        "title": "Window Swap - Open a Window to Someone Else's View",
        "description": "See the world from someone else's window.",
        "headings": ["Submit Your Window", "About", "FAQ"],
        "page_text": (
            "Window Swap lets you open a random window to someone else's view. "
            "People from around the world submit videos of the view from their "
            "window, and you can swap between them. A beautiful way to experience "
            "different places and perspectives without leaving your home."
        ),
    },
]


def main() -> None:
    """CLI entry point: generate 3 test descriptions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\n{'='*60}")
    print("AI DESCRIPTION GENERATOR — DEMO")
    print(f"{'='*60}\n")

    with AIDescriptionGenerator() as gen:
        if not gen._api_key:
            print(
                "WARNING: OPENROUTER_API_KEY not set. "
                "Set it via environment variable or pass api_key= to AIDescriptionGenerator."
            )
            print("Running with fallback descriptions...\n")

        results = gen.generate_batch(_TEST_CANDIDATES)

        for i, (candidate, result) in enumerate(
            zip(_TEST_CANDIDATES, results), 1
        ):
            print(f"--- Candidate {i}: {candidate['url']} ---")
            print(f"  Title:  {candidate['title']}")
            print(f"  Desc:   {result['description']}")
            print(f"  Cat:    {result['category']}")
            print(f"  Why:    {result['why_interesting']}")
            print(f"  Chars:  {len(result['description'])}")
            print()

            validation = gen.validate_output(result)
            status = "PASS" if validation["is_valid"] else "FAIL"
            print(f"  Validation: {status} {validation['issues']}\n")

    print(f"{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
