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
    CODEX_ACCESS_TOKEN  Optional override for the default openai-codex provider.
                        Otherwise reads Codex auth from ~/.hermes/auth.json.
    ANTHROPIC_API_KEY   Required only for --provider anthropic.
    OPENAI_API_KEY      Required only for --provider openai.
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
        "--provider", choices=["anthropic", "openai", "openai-codex"], default="openai-codex",
        help="LLM provider (default: openai-codex)",
    )
    parser.add_argument(
        "--base-url", metavar="URL", default=None,
        help="Custom base URL for OpenAI-compatible endpoints (e.g. local gateway)",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Log the estimate to MacroTrack API (requires MACROTRACK_BASE_URL + MACROTRACK_API_KEY)",
    )
    parser.add_argument(
        "--meal-type", choices=["breakfast", "lunch", "dinner", "snack"],
        default=None,
        help="Meal type for logging (default: inferred from local time)",
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

    # ── Import package ─────────────────────────────────────────────────────────
    # Support running from the repo root (calorie_estimator/ lives alongside run.py)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from calorie_estimator import CalorieEstimator
    except ImportError as e:
        print(f"Error: could not import calorie_estimator — {e}", file=sys.stderr)
        print("Install dependencies: pip install anthropic httpx pydantic openai", file=sys.stderr)
        sys.exit(1)

    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if provider == "openai-codex" and not CalorieEstimator.codex_auth_available():
        print(
            "Error: openai-codex requires CODEX_ACCESS_TOKEN or providers.openai-codex.tokens.access_token in Hermes auth.json.",
            file=sys.stderr,
        )
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

    # ── Log to MacroTrack ───────────────────────────────────────────────────────
    if args.log:
        await _log_to_macrotrack(result, args)


def _infer_meal_type() -> str:
    """Infer meal type from local hour."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 10:
        return "breakfast"
    elif 10 <= hour < 15:
        return "lunch"
    elif 15 <= hour < 18:
        return "snack"
    else:
        return "dinner"


async def _log_to_macrotrack(result, args) -> None:
    """POST the meal estimate to the MacroTrack API."""
    import json as _json
    import urllib.request as _req

    base_url = os.environ.get("MACROTRACK_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("MACROTRACK_API_KEY", "")

    if not base_url or not api_key:
        print("\n⚠️  MACROTRACK_BASE_URL or MACROTRACK_API_KEY not set — skipping log.", file=sys.stderr)
        return

    include_hidden = not args.no_hidden
    total = result.total_with_hidden if include_hidden else result.total
    meal_type = args.meal_type or _infer_meal_type()

    # Build a readable description from item names
    item_names = ", ".join(i.name for i in result.items) if result.items else (args.description or "meal")
    if args.description:
        description = f"{args.description} ({item_names})" if item_names not in args.description else args.description
    else:
        description = item_names

    from datetime import datetime, timezone
    logged_at = datetime.now(timezone.utc).isoformat()

    payload = _json.dumps({
        "calories": round(total.calories),
        "protein":  round(total.protein_g, 1),
        "carbs":    round(total.carbs_g, 1),
        "fat":      round(total.fat_g, 1),
        "description": description[:200],
        "logged_at": logged_at,
        "meal_type": meal_type,
    }).encode()

    req = _req.Request(
        f"{base_url}/api/food",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with _req.urlopen(req, timeout=10) as resp:
            logged = _json.loads(resp.read())
    except Exception as e:
        print(f"\n⚠️  MacroTrack log failed: {e}", file=sys.stderr)
        return

    # Fetch today's stats for remaining budget
    stats_req = _req.Request(
        f"{base_url}/api/stats/today",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with _req.urlopen(stats_req, timeout=10) as resp:
            stats = _json.loads(resp.read())
        rem = stats.get("remaining", {})
        intake = stats.get("intake", {})
        print(f"\n✅  Logged to MacroTrack — {meal_type}")
        print(f"   Today so far:  {intake.get('calories', 0)} kcal | "
              f"P {intake.get('protein', 0):.0f}g  "
              f"C {intake.get('carbs', 0):.0f}g  "
              f"F {intake.get('fat', 0):.0f}g")
        print(f"   Remaining:     {rem.get('calories', '?')} kcal | "
              f"P {rem.get('protein', '?'):.0f}g  "
              f"C {rem.get('carbs', '?'):.0f}g  "
              f"F {rem.get('fat', '?'):.0f}g")
    except Exception:
        print(f"\n✅  Logged to MacroTrack (ID: {logged.get('id', '?')})")


if __name__ == "__main__":
    asyncio.run(main())
