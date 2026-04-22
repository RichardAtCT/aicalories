"""Tests for the Open Food Facts barcode shortcut.

The shortcut must only fire when OFF returns a *complete* nutrition
record — a hit with only calories populated would otherwise render as
a high-confidence packaged-food result with zero macros (the
``_parse_product`` helper coerces any missing nutriment field to 0.0).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running either via pytest from the repo root, or standalone as
# `python tests/test_openfoodfacts.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_estimator.models import (
    NutrientProfile,
    OpenFoodFactsProduct,
)
from calorie_estimator.openfoodfacts import _parse_product


def _product(nutriments: dict) -> OpenFoodFactsProduct | None:
    payload = {
        "product_name": "Test Product",
        "brands": "Acme",
        "serving_size": "40 g",
        "serving_quantity": 40,
        "nutriments": nutriments,
    }
    return _parse_product("0000000000000", payload)


def test_full_hit_is_usable() -> None:
    product = _product(
        {
            "energy-kcal_100g": 450,
            "proteins_100g": 10,
            "fat_100g": 15,
            "carbohydrates_100g": 65,
            "fiber_100g": 7,
            "sugars_100g": 20,
            "sodium_100g": 0.15,
            "saturated-fat_100g": 2.5,
        }
    )
    assert product is not None
    assert product.has_usable_nutrition() is True


def test_partial_hit_without_macros_is_rejected() -> None:
    """OFF sometimes publishes just ``energy-kcal_100g``. That must not
    short-circuit the pipeline — we'd otherwise return a shiny high-
    confidence result with 0 g protein / 0 g fat / 0 g carbs."""
    product = _product({"energy-kcal_100g": 250})
    assert product is not None
    # _parse_product silently coerces missing fields to 0.0; the gate
    # lives on the product model.
    assert product.nutrients_per_100g.calories == 250
    assert product.nutrients_per_100g.protein_g == 0
    assert product.nutrients_per_100g.fat_g == 0
    assert product.nutrients_per_100g.carbs_g == 0
    assert product.has_usable_nutrition() is False


def test_zero_calories_is_rejected() -> None:
    product = _product({})
    assert product is not None
    assert product.has_usable_nutrition() is False


def test_kj_fallback_is_converted() -> None:
    """When only kJ is reported, we convert to kcal — but still need
    macros before we trust the record."""
    product = _product(
        {
            "energy_100g": 418.4,  # 100 kcal
            "proteins_100g": 5,
            "fat_100g": 2,
            "carbohydrates_100g": 15,
        }
    )
    assert product is not None
    assert product.nutrients_per_100g.calories == 100.0
    assert product.has_usable_nutrition() is True


def test_model_level_gate_rejects_partial_profile() -> None:
    """Directly constructing a product with just calories must also fail
    the gate, not just the parse path."""
    product = OpenFoodFactsProduct(
        barcode="1",
        product_name="Partial",
        nutrients_per_100g=NutrientProfile(calories=300),
    )
    assert product.has_usable_nutrition() is False


if __name__ == "__main__":  # pragma: no cover
    # Allow `python tests/test_openfoodfacts.py` without pytest installed.
    import traceback

    tests = [
        test_full_hit_is_usable,
        test_partial_hit_without_macros_is_rejected,
        test_zero_calories_is_rejected,
        test_kj_fallback_is_converted,
        test_model_level_gate_rejects_partial_profile,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
