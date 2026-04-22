---
name: aicalories
description: Estimate calories and macronutrients from food photos using the aicalories CLI and log them to MacroTrack.
---

# AI Calories Skill

Use this when Richard sends a food photo or asks for calories/macros for a meal.

Primary workflow:
1. Read the image and any caption/context.
2. Run the `aicalories` estimator.
3. Log to MacroTrack when asked (default for "log this" requests).
4. Reply concisely with exact logged calories, remaining calories, and protein status.

MacroTrack is the source of truth for nutrition state. Do not maintain your own running totals.

## Current references

- Skill path: `~/.hermes/skills/richard-workflows/aicalories/`
- MacroTrack env: `~/.hermes/.env.macroagent`
- aicalories env: `~/.hermes/.env.aicalories`
- Cached MacroTrack guide: `~/.hermes/data/macrotrack-llm-cache.md`
- Live MacroTrack agent guide: `GET /llm`

Before changing this skill, refresh the live guide:

```bash
set -a; source ~/.hermes/.env.macroagent >/dev/null 2>&1; set +a
curl -s "$MACROTRACK_BASE_URL/llm"
```

If `/llm` differs from the cache or workflow notes here, update the skill immediately.

## MacroTrack rules from `/llm`

- Start nutrition conversations with `GET /api/stats/onboard-status`.
- If onboarding is incomplete, fix missing profile / targets / first weight before normal logging.
- `POST /api/food` accepts `calories`, `protein`, `carbs`, `fat`, optional `description`, optional `logged_at`, and `meal_type`.
- `POST /api/food` also accepts `protein_g`, `carbs_g`, `fat_g` aliases; if both forms are present, the `_g` values win.
- Before estimating macros for a new log, call `GET /api/food/frequent` once per session and cache it; when the normalized description matches a frequent food entry, reuse `most_recent` by default and prefer `median` if prior logs vary a lot.
- To correct a food entry, prefer `PATCH /api/food/:id` for partial changes.
- `PUT /api/targets` now also supports phased-cut fields: `deficit_weeks`, `maintenance_weeks`, and `cycle_start_date` (all three together or none).
- `GET /api/diet-phases/preview?weeks=N` returns upcoming deficit/maintenance windows for communicating the schedule.
- `GET /api/stats/today` now returns `continuous_deficit_weeks` plus a `phase` block. If `phase.rebound_expected = true`, proactively warn that a 1–2 kg glycogen/water rebound is normal at the start of maintenance.
- `POST /api/weight` accepts optional `body_fat_pct` and now returns `rebound_expected`; if true on the first maintenance weigh-in, contextualize the bump immediately as glycogen/water rather than fat.
- If `GET /api/stats/today` returns `goal.rate_capped_by_floor = true`, warn that the requested rate is being capped by the BMR floor and the projection may be optimistic.
- Use exact API-returned numbers in replies. Do not round aggressively.
- Never moralize food choices.

## Setup

Use `python3.13`.

Dependencies:

```bash
pip3.13 install anthropic httpx pydantic openai --break-system-packages
```

Optional local USDA DB build:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/scripts/build_db.py
```

## Environment

`~/.hermes/.env.macroagent`

```bash
MACROTRACK_BASE_URL=http://...
MACROTRACK_API_KEY=...
```

`~/.hermes/.env.aicalories`

```bash
ANTHROPIC_API_KEY=...
USDA_API_KEY=...
CALORIE_ESTIMATOR_MODEL=...
```

Codex may also use `CODEX_ACCESS_TOKEN` or Hermes `auth.json`.

## Standard workflow

### 1) Check onboarding status

```bash
set -a
source ~/.hermes/.env.macroagent
set +a
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/stats/onboard-status"
```

If `ready` is `true`, continue.
If `ready` is `false`, complete the missing setup in this order:
1. `PUT /api/profile`
2. `PUT /api/targets`
3. `POST /api/weight`

### 2) Find the image

Telegram images are usually saved under:

```bash
~/.hermes/image_cache/
```

If you need the latest image path:

```bash
IMG=$(find ~/.hermes/image_cache -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \) -print | tail -n 1)
```

Prefer the explicit image path from the chat when available.

### 3) Run the estimator

Load both env files first:

```bash
set -a
source ~/.hermes/.env.aicalories
source ~/.hermes/.env.macroagent
set +a
```

Estimate only:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py --image "$IMG"
```

Estimate with context:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py \
  --image "$IMG" \
  --description "grilled salmon with roasted vegetables, no sauce"
```

Estimate and log:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py \
  --image "$IMG" \
  --description "..." \
  --log
```

Estimate and log with meal type:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py \
  --image "$IMG" \
  --description "..." \
  --log --meal-type lunch
```

Preferred fallback/default vision provider for Richard:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py \
  --image "$IMG" \
  --provider openai-codex \
  --description "..." \
  --log
```

Compact Codex estimate:

```bash
python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py \
  --image "$IMG" \
  --provider openai-codex \
  --compact
```

## Manual MacroTrack calls

Use these when you need to log or correct data directly.

Log food:

```bash
set -a; source ~/.hermes/.env.macroagent >/dev/null 2>&1; set +a
curl -s -X POST \
  -H "Authorization: Bearer $MACROTRACK_API_KEY" \
  -H "Content-Type: application/json" \
  "$MACROTRACK_BASE_URL/api/food" \
  -d '{"calories":700,"protein":50,"carbs":46,"fat":36,"description":"2 salmon patties, oven-baked sweet potato fries, green salad","meal_type":"dinner"}'
```

Patch a food entry:

```bash
curl -s -X PATCH \
  -H "Authorization: Bearer $MACROTRACK_API_KEY" \
  -H "Content-Type: application/json" \
  "$MACROTRACK_BASE_URL/api/food/ENTRY_ID" \
  -d '{"calories":450}'
```

Log weight with optional body fat:

```bash
curl -s -X POST \
  -H "Authorization: Bearer $MACROTRACK_API_KEY" \
  -H "Content-Type: application/json" \
  "$MACROTRACK_BASE_URL/api/weight" \
  -d '{"weight_kg":80.7,"body_fat_pct":23.4,"logged_at":"2026-04-22"}'
```

Fetch today’s status:

```bash
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/stats/today"
```

Check adherence:

```bash
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/stats/adherence?days=30"
```

Get weekly summary:

```bash
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/stats/weekly"
```

Get frequent foods for repeat-meal reuse:

```bash
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/food/frequent?limit=20"
```

Preview upcoming diet phases:

```bash
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/diet-phases/preview?weeks=12"
```

## Response style

Keep replies short and operational.

After logging food, include:
1. Exact calories logged
2. Exact remaining calories for the day
3. Protein status
4. Any important API warning (for example BMR floor rate cap)

Good example:

`Logged 581 kcal lunch. You're at 1180 today, 66 left. Protein is 76g, so still 54g to go.`

Do not dump a full itemized breakdown unless Richard asks.
Do not say "about" or "roughly" once something has been logged to MacroTrack.

For weight logs, confirm the weight/body-fat entry and, if helpful, mention the smoothed trend — but do not overreact to single-day swings.
If the API indicates a maintenance-phase rebound (`phase.rebound_expected` on today stats or `rebound_expected` on a weight log), explicitly say the 1–2 kg bump is expected glycogen/water, not fat.

## Troubleshooting

- If Anthropic returns `401 authentication_error: invalid x-api-key`, switch to `--provider openai-codex`.
- When using `--provider openai-codex`, explicitly override the model if `CALORIE_ESTIMATOR_MODEL` is set to an Anthropic model in the environment. Safe pattern:
  `CALORIE_ESTIMATOR_MODEL=gpt-5.4 python3.13 ~/.hermes/skills/richard-workflows/aicalories/run.py ... --provider openai-codex`
- Codex requires a real `CODEX_ACCESS_TOKEN` or a valid Hermes `auth.json` token.
- ChatGPT Codex vision works best via the streamed Responses API path used by the CLI; avoid adding unsupported parameters.
- If the user runs with `--no-hidden`, MacroTrack logging must use `result.total` rather than `result.total_with_hidden`.
- If the API returns `unresolved_days`, resolve them via `PUT /api/day-status/:date` with `status` set to `"fasting"` or `"skipped"`.
- Call `GET /api/food/frequent` once per session before estimating a described meal so repeat foods can reuse prior macros instead of re-estimating from scratch.
- If Richard asks about diet breaks, plateaus, or long cuts, inspect `continuous_deficit_weeks` and `phase` from `GET /api/stats/today`; use `GET /api/diet-phases/preview` to explain the current or proposed cycle.
- If the user mentions creatine, MacroTrack supports `POST /api/supplements` tracking. This matters because creatine can temporarily shift scale weight without calories.
- For local barcode testing on macOS, install `zbar` with Homebrew and make sure the dynamic linker can see it. In practice this may require:
  `brew install zbar`
  `export DYLD_LIBRARY_PATH=/opt/homebrew/lib:${DYLD_LIBRARY_PATH:-}`
  before running barcode scans via `pyzbar`.
- If Richard asks to make barcode scanning "ready next time", do not stop at updating docs/skill text. Make the runtime turnkey by persisting the fix in the actual launch path and environment (for example patch `run.py` to auto-prepend `/opt/homebrew/lib` to `DYLD_LIBRARY_PATH` on macOS when present, and/or store the env var in `~/.hermes/.env.aicalories`), then verify with a real barcode smoke test.
- If a repo checkout fails under the system Python because of modern type syntax like `X | None`, use a Python 3.11+ venv for local tests instead of Python 3.9.

### Real barcode smoke test (reusable)

When validating the packaged-food Stage 0 path end-to-end, use a real barcode image and confirm all three layers:
1. `detect_barcodes(image)` returns the UPC/EAN
2. `OpenFoodFactsClient.lookup(barcode)` returns a product
3. `CalorieEstimator().estimate(image=...)` returns an item with `source="barcode"`

A known-good real barcode used successfully in testing:
- `5449000000996` — Coca-Cola Original Taste

Minimal local smoke-test pattern:

```bash
cd /path/to/aicalories
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt python-barcode
brew install zbar   # macOS, once
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:${DYLD_LIBRARY_PATH:-}
```

Then generate a real barcode image and test it:

```bash
python - <<'PY'
from barcode import EAN13
from barcode.writer import ImageWriter
from pathlib import Path
from calorie_estimator.barcode import detect_barcodes
from calorie_estimator.openfoodfacts import OpenFoodFactsClient
from calorie_estimator import CalorieEstimator
import asyncio

code = '5449000000996'
out = Path('/tmp/test-barcode-5449000000996')
EAN13(code, writer=ImageWriter()).save(str(out))
img = out.with_suffix('.png').read_bytes()

print('detected=', detect_barcodes(img))

async def main():
    product = await OpenFoodFactsClient().lookup(code)
    print('lookup_found=', product is not None)
    result = await CalorieEstimator().estimate(image=img, description='barcode smoke test')
    print('source=', result.items[0].source if result.items else None)
    print(result.format_summary())

asyncio.run(main())
PY
```

## Notes

- Typical estimate cost: roughly $0.02–0.04
- USDA-backed workflow is more accurate than LLM-only fallback
- `run.py` lives alongside the `calorie_estimator/` package in the skill directory
- Re-check `/llm` whenever backend behavior appears different or Richard says the backend changed
