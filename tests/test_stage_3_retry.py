"""Regression tests for Stage 3 retry + EstimatorTransientError.

Before this fix, a malformed Stage 3 LLM response (JSON + trailing prose)
would silently collapse into `matches_raw = []`, Stage 4 would produce an
empty MealEstimate, and the caller had no way to tell "vague description"
from "transient LLM glitch". Now:

- `_parse_json` tolerates trailing prose.
- If parsing still fails, `_stage_3_disambiguate` retries once with a
  stricter reminder.
- If the retry also fails, `EstimatorTransientError` is raised — never a
  silent empty result.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_estimator.estimator import CalorieEstimator, EstimatorTransientError
from calorie_estimator.models import FoodCategory


def _items_with_candidates() -> list[dict]:
    return [
        {
            "item_id": 1,
            "item_name": "milk",
            "cooking_method": "none",
            "state": "raw",
            "visible_additions": [],
            "estimated_weight_g": 150,
            "confidence_identification": 0.9,
            "confidence_portion": 0.8,
            "category": FoodCategory.OTHER.value,
            "candidates": [
                {
                    "fdc_id": 123,
                    "description": "Milk, low fat, fluid, 1% fat",
                    "calories_per_100g": 42.0,
                    "protein_per_100g": 3.4,
                    "fat_per_100g": 1.0,
                    "carbs_per_100g": 5.0,
                    "fiber_per_100g": 0.0,
                    "sugar_per_100g": 5.0,
                    "sodium_per_100g": 44.0,
                    "saturated_fat_per_100g": 0.6,
                },
            ],
        }
    ]


def _make_estimator(responses: list[str]) -> CalorieEstimator:
    est = CalorieEstimator.__new__(CalorieEstimator)  # skip __init__
    est.provider = "claude-code"
    est.model = "test"
    est.temperature = 0.0
    est.apply_bias_correction = True
    est.estimate_hidden_cals = False
    est.include_confidence_ranges = False
    est.bias_corrections = {}
    est.api_key = ""
    est.base_url = None
    est.usda = None
    est.off = None

    call_count = {"n": 0}

    async def fake_call_llm(*, system, user_text, image_b64=None, media_type=None):
        i = call_count["n"]
        call_count["n"] += 1
        return responses[min(i, len(responses) - 1)]

    est._call_llm = fake_call_llm  # type: ignore[assignment]
    est._call_count = call_count  # type: ignore[attr-defined]
    return est


def test_trailing_prose_parses_on_first_attempt():
    """The hardened _parse_json should make this succeed without a retry."""
    response = (
        '{"matches": [{"item_id": 1, "selected_fdc_id": 123, '
        '"adjusted_weight_g": 150}]}\n\n'
        "Picked the 1% variant based on the description."
    )
    est = _make_estimator([response])
    matches = asyncio.run(
        est._stage_3_disambiguate(None, None, _items_with_candidates(), "150ml milk")
    )
    assert len(matches) == 1
    assert matches[0].selected_fdc_id == 123
    assert est._call_count["n"] == 1  # no retry needed


def test_unparseable_then_valid_retries():
    """First response is garbage; retry with stricter reminder returns valid JSON."""
    est = _make_estimator(
        [
            "definitely not JSON at all",
            '{"matches": [{"item_id": 1, "selected_fdc_id": 123, "adjusted_weight_g": 150}]}',
        ]
    )
    matches = asyncio.run(
        est._stage_3_disambiguate(None, None, _items_with_candidates(), "150ml milk")
    )
    assert len(matches) == 1
    assert est._call_count["n"] == 2  # retried once


def test_persistent_failure_raises_transient_error():
    """Two unparseable responses → EstimatorTransientError, never a silent empty."""
    est = _make_estimator(["nope", "still nope"])
    with pytest.raises(EstimatorTransientError):
        asyncio.run(
            est._stage_3_disambiguate(
                None, None, _items_with_candidates(), "150ml milk"
            )
        )
    assert est._call_count["n"] == 2
