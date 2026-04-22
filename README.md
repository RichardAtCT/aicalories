# Calorie Estimator — Agent Tool

A self-contained Python tool that estimates calories and macronutrients from food photos (+ optional text descriptions). Designed to be called by any agent framework — Telegram bots, Discord bots, API endpoints, CLI tools, etc.

**Live deployment:** http://w00kkgowkg4cco44coo440kw.37.27.242.72.sslip.io:3000

## Architecture

```
Agent (Telegram bot, etc.)
  │
  │  estimate(image_bytes, text_description?, barcode_hint?)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CalorieEstimator                                               │
│                                                                 │
│  Stage 0: BARCODE ──────────────────────────── (pyzbar + OFF)   │
│  │  Detect UPC/EAN → look up Open Food Facts                    │
│  │  • Hit with nutrition  → return immediately                   │
│  │  • Hit without data    → ask user for nutrition-label photo  │
│  │  • No barcode          → fall through to Stage 1             │
│  ▼                                                              │
│  Stage 1: VISUAL ANALYSIS ──────────────────── (LLM + vision)  │
│  │  Identify foods, cooking methods, reference objects          │
│  │  Estimate dimensions and weight in grams                     │
│  │  Assign confidence scores                                    │
│  ▼                                                              │
│  Stage 2: DATABASE LOOKUP ──────────────────── (USDA FDC API)  │
│  │  For each food item, retrieve candidate nutrition entries     │
│  │  Include cooking method + state in search query              │
│  ▼                                                              │
│  Stage 3: DISAMBIGUATION ───────────────────── (LLM, 2nd pass) │
│  │  Re-prompt with photo + candidates + user text               │
│  │  Select best database match per item                         │
│  │  Adjust weight estimates if needed                           │
│  ▼                                                              │
│  Stage 4: CALCULATION ──────────────────────── (deterministic)  │
│  │  nutrients = db_per_100g × (estimated_grams / 100)           │
│  │  Apply bias corrections per food category                    │
│  │  Aggregate meal totals                                       │
│  ▼                                                              │
│  Returns: MealEstimate (structured data)                        │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Agent formats response for user
```

### Packaged foods (barcode path)

If a UPC/EAN is visible in the photo, Stage 0 decodes it with `pyzbar` and
looks the product up in [Open Food Facts](https://world.openfoodfacts.org).
When OFF has full nutrition data we skip the LLM entirely and return a
high-confidence result built from the package's own values, annotated with
the serving size so you can adjust the amount if you ate more or less.

If the barcode isn't in OFF — or the product has no nutrition data — the
estimate comes back with a warning asking the user to photograph the
nutrition label (or the food itself). On the follow-up photo, pass the
stored barcode as `barcode_hint=...` to `estimate()`; the library reads the
label, returns a single-item `MealEstimate`, and attaches a
`pending_off_contribution`. After the user confirms, call
`estimator.submit_pending_contribution(...)` to push the data back to OFF
so the next person scanning that product gets it for free. The library
never submits on its own.

### Why this architecture?

The research is clear on three things:

1. **LLMs are good at identifying food and reasoning about portions, but bad at
   recalling precise nutrition values.** So we use the LLM for perception and the
   USDA database for numbers.

2. **Chain-of-thought prompting for volumetric estimation significantly improves
   accuracy.** Forcing the model through identify → measure dimensions → estimate
   weight (rather than just "guess the calories") reduces portion size errors.

3. **A disambiguation pass against database candidates improves food matching.**
   Cooking method alone (fried vs baked) can change calories by 30-50%. The second
   LLM call with real database options to choose from catches these distinctions.

## CLI

`run.py` is a standalone CLI wrapper — no server, no bot token needed.

```bash
# Basic usage
python run.py --image /path/to/food.jpg

# With a description for better accuracy
python run.py --image /path/to/food.jpg --description "fried in olive oil, large portion"

# From a URL
python run.py --url https://example.com/meal.jpg --description "pasta carbonara"

# Raw JSON output (for scripting / agent tools)
python run.py --image food.jpg --json

# One-line compact summary
python run.py --image food.jpg --compact

# Exclude hidden calorie estimates (cooking oil, butter, etc.)
python run.py --image food.jpg --no-hidden
```

**Output example:**
```
🍽️ **Meal Estimate**

**Grilled chicken breast, no skin** (185g) — 306 kcal  [●●●●○]
  P: 57g · F: 7g · C: 0g

**Brown rice, cooked** (220g) — 240 kcal  [●●●●○]
  P: 5g · F: 2g · C: 51g

_Estimated hidden calories:_
  +45 kcal — cooking oil (brushed on grill)

────────────────────────────────────
**TOTAL: 591 kcal**
Protein: 62g · Fat: 10g · Carbs: 51g · Fiber: 4g
```

## Setup

### Requirements

```
python >= 3.11
anthropic >= 0.40.0
httpx >= 0.27.0        # async HTTP for USDA API
pydantic >= 2.0        # data models
```

### Install

```bash
pip install -r requirements.txt

# Barcode scanning requires the zbar system library:
#   Debian/Ubuntu:  sudo apt-get install libzbar0
#   macOS:          brew install zbar
```

If zbar isn't installed the estimator still works — it just skips the barcode
stage and goes straight to visual analysis.

### Configuration

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-..."

# Recommended (free key — better accuracy via USDA database)
export USDA_API_KEY="your-usda-key"         # free at https://fdc.nal.usda.gov/api-key-signup

# Optional
export CALORIE_ESTIMATOR_MODEL="claude-sonnet-4-20250514"  # default

# Optional — attribute Open Food Facts contributions to a real account
# (anonymous edits work without these, logged against your IP).
export OFF_USERNAME="your-off-user"
export OFF_PASSWORD="your-off-password"
# Optional — point at the OFF staging server when testing submissions
# so you don't pollute production data.
export OFF_BASE_URL="https://world.openfoodfacts.net"
```

Get a free USDA API key at: https://fdc.nal.usda.gov/api-key-signup

Without a USDA key the estimator falls back to single-pass LLM estimation using a
bundled dataset of ~500 common foods. Accuracy is lower (~35-50% MAPE vs ~25-35%
with the full pipeline).

### Local Database (optional, recommended)

Build a local copy of the USDA FoodData Central database for offline use:

```bash
python scripts/build_db.py
```

This downloads three USDA datasets (~300 MB download) and builds a SQLite database
with FTS5 full-text search (~50 MB). One-time setup — no USDA API key needed after
that. The local DB is auto-detected; if absent, the estimator falls back to the live
API as before.

## Quick Start

```python
from calorie_estimator import CalorieEstimator

estimator = CalorieEstimator()

# From a file
with open("lunch.jpg", "rb") as f:
    result = await estimator.estimate(
        image=f.read(),
        description="Chicken stir-fry with rice, cooked in sesame oil"
    )

print(result.total.calories)        # 685.2
print(result.total.protein_g)       # 42.1
print(result.format_summary())      # formatted text for the user

# result.items gives you per-item breakdown
for item in result.items:
    print(f"{item.name}: {item.nutrients.calories} kcal ({item.weight_g}g)")
```

## Integration Examples

### Telegram Bot

See [examples/telegram_bot.py](examples/telegram_bot.py) for a full working example.

```python
# Minimal Telegram integration
from calorie_estimator import CalorieEstimator

estimator = CalorieEstimator()

async def handle_photo(update, context):
    photo = await update.message.photo[-1].get_file()
    image_bytes = await photo.download_as_bytearray()

    caption = update.message.caption or ""
    result = await estimator.estimate(image=bytes(image_bytes), description=caption)

    await update.message.reply_text(result.format_summary())
```

### As an Agent Tool (e.g. LangChain / Claude tool_use)

```python
# Tool definition for any agent framework
CALORIE_TOOL = {
    "name": "estimate_calories",
    "description": "Estimate calories and macronutrients from a food photo. "
                   "Returns per-item and total nutritional breakdown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "image_base64": {
                "type": "string",
                "description": "Base64-encoded food image (JPEG or PNG)"
            },
            "description": {
                "type": "string",
                "description": "Optional text description of the meal, "
                               "cooking method, or portion context"
            }
        },
        "required": ["image_base64"]
    }
}
```

### As a FastAPI Endpoint

```python
from fastapi import FastAPI, UploadFile, Form
from calorie_estimator import CalorieEstimator

app = FastAPI()
estimator = CalorieEstimator()

@app.post("/estimate")
async def estimate(image: UploadFile, description: str = Form("")):
    image_bytes = await image.read()
    result = await estimator.estimate(image=image_bytes, description=description)
    return result.model_dump()
```

## Configuration Options

```python
estimator = CalorieEstimator(
    # LLM provider — "anthropic" (default), "openai", or "custom"
    provider="anthropic",

    # Model to use for vision analysis
    model="claude-sonnet-4-20250514",

    # USDA API key (or set USDA_API_KEY env var)
    usda_api_key="your-key",

    # Apply empirical bias corrections (recommended)
    apply_bias_correction=True,

    # Include hidden calorie estimates (cooking oil, etc.)
    estimate_hidden_calories=True,

    # Return confidence ranges for low-confidence items
    include_confidence_ranges=True,

    # Temperature for LLM calls (lower = more deterministic)
    temperature=0.1,

    # Custom correction factors (override defaults)
    bias_corrections=None,
)
```

## Accuracy Notes

Based on published research (PMC 2025, DietAI24, Macroscanner):

- **Expected MAPE**: ~25-35% for calories on typical meals
- **Best case**: Simple, single-item meals with visible reference objects (~15-20%)
- **Worst case**: Complex mixed dishes, sauces, hidden oils (~40-50%)
- **Systematic bias**: LLMs tend to underestimate large portions and calorie-dense
  foods (nuts, oils, sauces). The bias correction module addresses this.

The text description input is your biggest accuracy lever — cooking method, oil
used, sugar added, and portion context are invisible to the camera but can shift
calories by 30%+.

## Cost Per Estimate

| Component | Approximate Cost |
|-----------|-----------------|
| Stage 1: Vision analysis | ~$0.01-0.02 |
| Stage 3: Disambiguation | ~$0.01-0.02 |
| USDA API | Free (1,000 req/hr) |
| **Total** | **~$0.02-0.04** |

## Project Structure

```
aicalories/
├── README.md                          # this file
├── requirements.txt
├── run.py                             # CLI entry point
├── calorie_estimator/
│   ├── __init__.py                    # public API
│   ├── estimator.py                   # main orchestrator
│   ├── prompts.py                     # all LLM prompts (Stage 1 & 3)
│   ├── usda.py                        # USDA FoodData Central client (local DB + API)
│   ├── models.py                      # Pydantic data models
│   └── corrections.py                 # bias corrections & hidden calorie heuristics
├── scripts/
│   └── build_db.py                    # build local USDA SQLite database
├── data/
│   ├── common_foods.json              # fallback nutrition data (top 500 foods)
│   └── usda.db                        # local USDA database (git-ignored, built by scripts/build_db.py)
└── telegram_bot.py                    # example Telegram bot integration
```
