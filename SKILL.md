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

**Environment:** `~/.openclaw/workspace/.env.aicalories`
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

### 1. Get the image path

When Richard sends a photo via Telegram, OpenClaw downloads it to:
```
~/.openclaw/workspace/tmp/img-NNN.jpg
```

Get the most recently saved image:
```bash
FOOD_IMG=$(ls -t ~/.openclaw/workspace/tmp/img-*.jpg 2>/dev/null | head -1)
```

### 2. Run the estimator

```bash
# Load env vars
set -a; source ~/.openclaw/workspace/.env.aicalories; set +a

# Basic (no description)
python3.13 ~/.openclaw/skills/aicalories/run.py --image "$FOOD_IMG"

# With a description (better accuracy — use any caption Richard provided)
python3.13 ~/.openclaw/skills/aicalories/run.py \
  --image "$FOOD_IMG" \
  --description "grilled salmon with roasted vegetables, no sauce"

# Compact one-liner (useful for quick replies)
python3.13 ~/.openclaw/skills/aicalories/run.py --image "$FOOD_IMG" --compact
```

### 3. Reply to Richard

Format the output naturally:
- Lead with the total calories in bold
- Include per-item breakdown
- Mention confidence if low
- Note hidden calories (oil, butter, etc.) if present
- Keep it concise — Richard doesn't need the raw JSON

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
