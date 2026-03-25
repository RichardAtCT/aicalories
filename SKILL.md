---
name: aicalories
description: Estimate calories and macronutrients from food photos using a 4-stage AI pipeline (vision → USDA lookup → disambiguation → calculation). Use when Richard sends a food photo or asks about calories in a meal.
---

# AI Calories Skill

Estimate calories and macronutrients from food photos. Uses a 4-stage pipeline:
visual analysis → USDA database lookup → LLM disambiguation → deterministic calculation.

## Trigger

Use this skill when:
- Richard sends a food photo (with or without a caption)
- Richard asks "how many calories is this?" alongside an image
- Richard asks for a calorie/macro breakdown of food he's about to eat or just ate

## Setup

**Location:** `~/.openclaw/skills/aicalories/`

**Python:** Use `python3.13` (code uses `X | Y` union syntax requiring Python 3.10+)

**Dependencies:**
```bash
pip3.13 install anthropic httpx pydantic openai --break-system-packages
```

**Local USDA database (optional, for offline use):**
```bash
python3.13 ~/.openclaw/skills/aicalories/scripts/build_db.py
```
This builds a local SQLite database from USDA FoodData Central (~300 MB download, ~50 MB DB). Once built, the estimator uses it automatically — no `USDA_API_KEY` required.

**MacroTrack env:** `~/.openclaw/workspace/.env.macroagent`
```
MACROTRACK_BASE_URL=http://w00kkgowkg4cco44coo440kw.37.27.242.72.sslip.io:3000
MACROTRACK_API_KEY=<key>
```

**Aicalories env:** `~/.openclaw/workspace/.env.aicalories`
```bash
# Anthropic OAuth token (from ~/.openclaw/agents/main/agent/auth-profiles.json)
ANTHROPIC_API_KEY=<sk-ant-oat-...token...>
USDA_API_KEY=<your-usda-key>
CALORIE_ESTIMATOR_MODEL=claude-haiku-4-5
```

To extract the Anthropic token:
```bash
cat ~/.openclaw/agents/main/agent/auth-profiles.json | python3.13 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['profiles']['anthropic:default']['token'])"
```

## How to Use

### 0. Check onboarding status (start of any nutrition conversation)

```bash
set -a; source ~/.openclaw/workspace/.env.macroagent; set +a
curl -s -H "Authorization: Bearer $MACROTRACK_API_KEY" "$MACROTRACK_BASE_URL/api/onboard-status"
```

If `ready` is `false`, complete what's missing before logging food (profile → targets → first weight).
If `ready` is `true`, skip straight to the task.

### 1. Get the image path

When Richard sends a photo via Telegram, OpenClaw downloads it to:
```
~/.openclaw/workspace/tmp/img-NNN.jpg
```

Get the most recently saved image:
```bash
FOOD_IMG=$(ls -t ~/.openclaw/workspace/tmp/img-*.jpg 2>/dev/null | head -1)
```

### 2. Run the estimator + log to MacroTrack

```bash
# Load both env files
set -a
source ~/.openclaw/workspace/.env.aicalories
source ~/.openclaw/workspace/.env.macroagent
set +a

# Estimate + auto-log to MacroTrack (default for food photos)
python3.13 ~/.openclaw/skills/aicalories/run.py \
  --image "$FOOD_IMG" \
  --description "grilled salmon with roasted vegetables, no sauce" \
  --log

# With explicit meal type
python3.13 ~/.openclaw/skills/aicalories/run.py \
  --image "$FOOD_IMG" \
  --description "..." \
  --log --meal-type lunch

# Estimate only (no logging)
python3.13 ~/.openclaw/skills/aicalories/run.py --image "$FOOD_IMG"
```

### 3. Reply to Richard

After `--log`, the script prints:
- Full macro breakdown per item
- `✅ Logged to MacroTrack — lunch`
- Today's running total and **remaining budget**

Format your reply as:
- Lead with total calories + remaining budget
- Brief per-item breakdown
- Protein status (is he on track for his 165g daily target?)
- Keep it under 5 lines — no walls of text

**Example:**
> 🍽️ **581 kcal logged** · 1,646 kcal remaining
> - Chicken thigh (194g) · Rice (155g) · Broccoli (90g) · Honey mustard sauce
> Protein: 45g today, 120g still to go — front-load dinner if you can.

**Example reply:**
> 🍽️ **~590 kcal**
> 
> - Grilled chicken breast (185g) — 306 kcal · P 57g F 7g C 0g
> - Brown rice (220g) — 240 kcal · P 5g F 2g C 51g
> - +45 kcal estimated cooking oil
> 
> **Total: ~591 kcal** | Protein 62g · Fat 10g · Carbs 51g · Fiber 4g

## Tips for Better Accuracy

- Encourage Richard to add a caption with cooking details: _"fried in butter"_, _"large portion"_, _"restaurant-size"_
- The description is the single biggest accuracy lever — can shift estimates by 30%+
- Low confidence items show a ↕ range — mention this if it's a big spread

## Notes

- Typical cost per estimate: ~$0.02–0.04 (two Anthropic API calls + USDA queries)
- USDA key gives full 4-stage pipeline (~25-35% MAPE); without it falls back to LLM-only (~35-50% MAPE)
- `run.py` is in the repo root alongside the `calorie_estimator/` package
