#!/usr/bin/env python3
"""CLI entry point for CalorieEstimator.

Estimates calories and macronutrients from a food photo.

Usage:
    python run.py --image /path/to/food.jpg
    python run.py --image /path/to/food.jpg --description "fried in olive oil, large portion"
    python run.py --url https://example.com/food.jpg
    python run.py --image food.jpg --json            # raw JSON output
    python run.py --image food.jpg --no-hidden       # exclude hidden calorie estimates

Environment variables:
    ANTHROPIC_API_KEY   Required. Anthropic API key.
    USDA_API_KEY        Optional. Free key from https://fdc.nal.usda.gov/api-key-signup
                        Falls back to bundled food database if not set.
"""

import argparse
import asyncio
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate calories and macronutrients from a food photo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image", "-i", metavar="PATH",
        help="Local image file path (JPEG, PNG, WebP, or GIF)",
    )
    source.add_argument(
        "--url", "-u", metavar="URL",
        help="Image URL — downloaded automatically to a temp file",
    )

    parser.add_argument(
        "--description", "-d", metavar="TEXT", default="",
        help="Optional text description: cooking method, portion context, hidden ingredients",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--no-hidden", action="store_true",
        help="Exclude hidden calorie estimates (cooking oil, butter, etc.)",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="One-line summary (overrides --json)",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"], default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--base-url", metavar="URL", default=None,
        help="Custom base URL for OpenAI-compatible endpoints (e.g. local gateway)",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # ── Resolve image to bytes ─────────────────────────────────────────────────
    tmp_path = None

    if args.url:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            tmp_path = tmp.name
            print(f"Downloading image from URL…", file=sys.stderr)
            urllib.request.urlretrieve(args.url, tmp_path)
            image_path = tmp_path
        except Exception as e:
            print(f"Error: could not download image — {e}", file=sys.stderr)
            sys.exit(1)
    else:
        image_path = args.image

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: image not found — {image_path}", file=sys.stderr)
        sys.exit(1)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Check API key ──────────────────────────────────────────────────────────
    provider = args.provider
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # ── Import package ─────────────────────────────────────────────────────────
    # Support running from the repo root (calorie_estimator/ lives alongside run.py)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from calorie_estimator import CalorieEstimator
    except ImportError as e:
        print(f"Error: could not import calorie_estimator — {e}", file=sys.stderr)
        print("Install dependencies: pip install anthropic httpx pydantic", file=sys.stderr)
        sys.exit(1)

    # ── Run estimator ──────────────────────────────────────────────────────────
    estimator = CalorieEstimator(
        provider=args.provider,
        base_url=args.base_url,
        apply_bias_correction=True,
        estimate_hidden_cals=not args.no_hidden,
        include_confidence_ranges=True,
    )

    try:
        result = await estimator.estimate(
            image=image_bytes,
            description=args.description or None,
        )
    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Output ─────────────────────────────────────────────────────────────────
    if args.compact:
        print(result.format_compact())
    elif args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(result.format_summary(include_hidden=not args.no_hidden))


if __name__ == "__main__":
    asyncio.run(main())
