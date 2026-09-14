"""Main entry point and CLI for the Daily Random Website pipeline.

Usage:
    python -m src.main [OPTIONS]

Options:
    --dry-run               Execute pipeline without persisting posts or screenshots
    --source SOURCE         Source to collect from (hackernews, github, rss, wikipedia, awesome_lists, or all)
    --limit N               Number of posts to discover and publish (default: 1)
    --output-dir DIR        Directory for generated static site (default: dist)
    --check-credits         Check Netlify deployment credit/quota status and exit
    --test-error-handling   Simulate Hacker News failure to verify fallback and DLQ
    -v, --verbose           Enable debug-level logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import ContentPipeline
from src.resilience import check_netlify_credits


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for pipeline execution."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Clear existing handlers to prevent duplicate output
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Daily Random Website Autonomous Discovery Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without modifying database/posts or generating permanent artifacts",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["hackernews", "github", "rss", "wikipedia", "awesome_lists", "all"],
        help="Limit discovery to a specific collector source",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of new posts to discover and feature",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dist",
        help="Target directory for static HTML/CSS/JS output",
    )

    parser.add_argument(
        "--check-credits",
        action="store_true",
        help="Verify Netlify credit/quota status and exit",
    )

    parser.add_argument(
        "--test-error-handling",
        action="store_true",
        help="Simulate primary source (Hacker News) failure to verify fallback sources and DLQ",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main execution function returning exit code."""
    opts = parse_args(args)
    setup_logging(opts.verbose)
    logger = logging.getLogger("src.main")

    logger.info("Initializing Daily Random Website Autonomous Pipeline")

    # 1. Credit check mode
    if opts.check_credits:
        logger.info("Checking Netlify deployment credits and status...")
        status = check_netlify_credits()
        logger.info("Netlify credit status: %s (safe_to_deploy=%s)", status["status"], status["credits_available"])
        logger.info("Details: %s", status["message"])
        return 0 if status["credits_available"] else 1

    # 2. Run Pipeline
    pipeline = ContentPipeline(output_dir=Path(opts.output_dir))

    if opts.test_error_handling:
        logger.info("Running under --test-error-handling mode: Simulating Hacker News failure")

    result = pipeline.run(
        dry_run=opts.dry_run,
        source=opts.source,
        limit=opts.limit,
        simulate_hn_failure=opts.test_error_handling,
    )

    # 3. Summary Reporting
    logger.info("================ PIPELINE SUMMARY ================")
    logger.info("Success:          %s", result["success"])
    logger.info("Dry run:          %s", result["dry_run"])
    logger.info("Candidates:       %d collected", result["collected_count"])
    logger.info("Deduplicated:     %d remaining", result["filtered_count"])
    logger.info("Scored:           %d candidates", result["scored_count"])
    logger.info("Published posts:  %d posts", len(result["published_posts"]))
    logger.info("Site generated:   %s", result["site_generated"])
    if result["errors"]:
        logger.warning("Pipeline reported %d error(s):", len(result["errors"]))
        for err in result["errors"]:
            logger.warning("  - %s", err)
    logger.info("==================================================")

    if not result["success"] and not opts.dry_run:
        logger.error("Pipeline finished with errors or no posts published.")
        return 1

    logger.info("Pipeline run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
