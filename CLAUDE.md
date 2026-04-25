# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Calories is a Python library that estimates calories and macronutrients from food photos. It uses a 4-stage pipeline: Visual Analysis (LLM vision) → USDA Database Lookup → LLM Disambiguation → Deterministic Calculation. The key insight is using LLMs for perception/reasoning but the USDA database for actual nutrition values.

## Setup

```bash
pip install anthropic httpx pydantic
```

Required env vars:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export USDA_API_KEY="..."  # free at https://fdc.nal.usda.gov/api-key-signup
```

## Running

```python
from calorie_estimator import CalorieEstimator
estimator = CalorieEstimator()
result = await estimator.estimate(image=image_bytes, description="optional text")
```

There is no test suite, build system, or linting configuration. The project is a flat Python module structure (no package directory despite README suggesting one).

## Architecture

- **estimator.py** — `CalorieEstimator` class orchestrating the 4-stage pipeline. Handles LLM provider abstraction (Anthropic default, OpenAI fallback) and JSON parsing from LLM outputs.
- **models.py** — Pydantic v2 models: `FoodItem`, `VisualAnalysis`, `USDACandidate`, `FoodMatch`, `NutrientProfile` (supports arithmetic), `MealEstimate` (has `format_summary()`).
- **prompts.py** — System/user prompts for Stage 1 (chain-of-thought visual analysis), Stage 3 (disambiguation against USDA candidates), and `TEXT_EXTRACTION_SYSTEM` for the text-only Stage 1. Also has `FALLBACK_SYSTEM` used by both single-pass fallbacks: image path when the USDA API is down (`_fallback_estimate`) and text path when Stage 2 returns no candidates (`_fallback_estimate_from_text`).
- **usda.py** — Async `USDAClient` using `httpx`. Searches FDC v1 API, prefers Survey (FNDDS) data types. Falls back to bundled `common_foods.json`.
- **corrections.py** — Per-category weight bias multipliers (e.g., grains +15%, sauces +30%) and hidden calorie heuristics (cooking oil, butter, dressing).
- **telegram_bot.py** — Example Telegram bot integration.

## Key Design Decisions

- Async-first (`async/await` throughout, `httpx.AsyncClient`)
- Two LLM passes: perception first, then disambiguation against real DB entries
- Low temperature (0.1) for deterministic outputs
- Graceful degradation: if Stage 2 returns no candidates (text path) or the USDA API fails (image path), falls back to a single-pass LLM estimate. The fallback prepends `"LLM-estimated nutrition (item not in USDA data) — less precise"` to `warnings`, caps `overall_confidence` at 0.6, and never returns a fully empty `MealEstimate` for a reasonable description (so callers don't have to invent their own clarifier).
- Defensive enum coercion: `_normalize_enums` maps `FoodCategory` singulars and synonyms (`beverage` → `beverages`, `veg` → `vegetables`, unknown → `other`) so the LLM occasionally returning the singular form doesn't silently void Stage 1 via Pydantic validation failure.
- `NutrientProfile` supports `+` operator and `.scale()` for aggregation
