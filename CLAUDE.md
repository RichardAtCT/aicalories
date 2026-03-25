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
- **prompts.py** — System/user prompts for Stage 1 (chain-of-thought visual analysis) and Stage 3 (disambiguation against USDA candidates). Also has `FALLBACK_SYSTEM` for single-pass mode.
- **usda.py** — Async `USDAClient` using `httpx`. Searches FDC v1 API, prefers Survey (FNDDS) data types. Falls back to bundled `common_foods.json`.
- **corrections.py** — Per-category weight bias multipliers (e.g., grains +15%, sauces +30%) and hidden calorie heuristics (cooking oil, butter, dressing).
- **telegram_bot.py** — Example Telegram bot integration.

## Key Design Decisions

- Async-first (`async/await` throughout, `httpx.AsyncClient`)
- Two LLM passes: perception first, then disambiguation against real DB entries
- Low temperature (0.1) for deterministic outputs
- Graceful degradation: if USDA API fails, falls back to single-pass LLM estimation
- `NutrientProfile` supports `+` operator and `.scale()` for aggregation
