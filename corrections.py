"""Bias corrections and hidden calorie heuristics.

LLMs systematically underestimate certain food categories and miss hidden
calorie sources. These corrections are derived from published validation
studies and should be calibrated against your own evaluation dataset.

Sources:
- PMC 2025: LLM nutrition estimation validation (ChatGPT, Claude, Gemini)
- DietAI24: MLLM + RAG framework evaluation
- Macroscanner: practical LLM food analysis observations
"""

from __future__ import annotations

from .models import (
    FoodCategory,
    FoodItem,
    FoodMatch,
    HiddenCalorieEstimate,
)

# ── Systematic Bias Corrections ──────────────────────────────
#
# LLMs tend to underestimate weight for calorie-dense and starchy foods,
# and slightly overestimate for leafy/voluminous foods.
#
# These factors are multiplied against the LLM's weight estimate.
# Values > 1.0 = LLM typically underestimates (correct upward)
# Values < 1.0 = LLM typically overestimates (correct downward)
#
# CALIBRATE THESE against your own evaluation dataset.

DEFAULT_BIAS_CORRECTIONS: dict[FoodCategory, float] = {
    FoodCategory.PROTEINS: 1.05,
    FoodCategory.GRAINS_STARCHES: 1.15,      # rice, pasta, bread — consistently underestimated
    FoodCategory.VEGETABLES: 0.95,
    FoodCategory.FRUITS: 1.00,
    FoodCategory.DAIRY: 1.05,
    FoodCategory.SAUCES_DRESSINGS: 1.30,     # biggest blind spot
    FoodCategory.BEVERAGES: 1.00,            # usually container-based, accurate
    FoodCategory.NUTS_SEEDS: 1.20,           # dense, small, underestimated
    FoodCategory.OILS_FATS: 1.25,            # hard to see quantity
    FoodCategory.SWEETS_DESSERTS: 1.10,
    FoodCategory.MIXED_DISHES: 1.10,
    FoodCategory.OTHER: 1.05,
}


def apply_weight_correction(
    weight_g: float,
    category: FoodCategory,
    corrections: dict[FoodCategory, float] | None = None,
) -> float:
    """Apply bias correction to an estimated weight.

    Args:
        weight_g: LLM's estimated weight in grams.
        category: Food category for lookup.
        corrections: Custom correction factors. Defaults to published baselines.

    Returns:
        Corrected weight in grams.
    """
    factors = corrections or DEFAULT_BIAS_CORRECTIONS
    factor = factors.get(category, 1.0)
    return round(weight_g * factor, 1)


# ── Hidden Calorie Estimation ────────────────────────────────
#
# Things the camera cannot see but significantly affect calorie count.
# These are applied as additive estimates based on visible cooking cues.


def estimate_hidden_calories(
    items: list[FoodItem] | list[FoodMatch],
    user_description: str | None = None,
) -> list[HiddenCalorieEstimate]:
    """Estimate hidden calories based on food items and cooking methods.

    Looks for cooking methods and food types that imply hidden calorie
    sources (cooking oil, butter, sugar, etc.) that aren't visible.

    Args:
        items: Food items from Stage 1 or Stage 3.
        user_description: Optional user text that may clarify hidden ingredients.

    Returns:
        List of hidden calorie estimates.
    """
    hidden = []
    desc_lower = (user_description or "").lower()

    # Track what we've already accounted for
    has_fried = False
    has_sauteed = False
    has_salad = False
    has_bread_toast = False
    has_pasta = False
    has_rice_with_topping = False

    for item in items:
        name = _get_name(item).lower()
        method = _get_method(item).lower()
        additions = _get_additions(item)
        additions_lower = [a.lower() for a in additions]

        # ── Fried foods: assume cooking oil
        if "fried" in method or "fried" in name or "deep-fried" in name:
            if not has_fried:
                has_fried = True
                # Shallow fry ≈ 1 tbsp absorbed, deep fry ≈ 2 tbsp
                is_deep = "deep" in method or "deep" in name
                oil_tbsp = 2.0 if is_deep else 1.0
                hidden.append(HiddenCalorieEstimate(
                    source="cooking oil (absorbed from frying)",
                    estimated_calories=round(oil_tbsp * 120, 0),
                    confidence=0.4,
                    note=f"~{oil_tbsp:.0f} tbsp oil absorbed",
                ))

        # ── Sautéed / stir-fried: assume some oil
        elif "sauté" in method or "saute" in method or "stir" in method:
            if not has_sauteed:
                has_sauteed = True
                hidden.append(HiddenCalorieEstimate(
                    source="cooking oil (sauté/stir-fry)",
                    estimated_calories=90,
                    confidence=0.35,
                    note="~0.75 tbsp oil estimated",
                ))

        # ── Salad without visible dressing → assume some dressing
        if "salad" in name and not any(
            d in additions_lower
            for d in ["dressing", "vinaigrette", "ranch", "caesar", "oil"]
        ):
            if not has_salad:
                has_salad = True
                # Unless user says "no dressing"
                if "no dressing" not in desc_lower and "undressed" not in desc_lower:
                    hidden.append(HiddenCalorieEstimate(
                        source="salad dressing (estimated)",
                        estimated_calories=140,
                        confidence=0.25,
                        note="~2 tbsp standard dressing assumed; "
                             "tell me if none was used",
                    ))

        # ── Bread/toast → assume butter unless stated otherwise
        if any(w in name for w in ["bread", "toast", "roll", "bun"]):
            if not has_bread_toast:
                has_bread_toast = True
                if "no butter" not in desc_lower and "dry" not in desc_lower:
                    if not any(
                        b in additions_lower for b in ["butter", "margarine", "jam"]
                    ):
                        hidden.append(HiddenCalorieEstimate(
                            source="butter on bread/toast (estimated)",
                            estimated_calories=70,
                            confidence=0.2,
                            note="~2 tsp butter assumed; tell me if none was used",
                        ))

        # ── Pasta → check for sauce
        if any(w in name for w in ["pasta", "spaghetti", "penne", "linguine"]):
            if not has_pasta:
                has_pasta = True
                # If no sauce is explicitly listed in items, assume some
                all_names = " ".join(_get_name(i).lower() for i in items)
                if "sauce" not in all_names and "pesto" not in all_names:
                    hidden.append(HiddenCalorieEstimate(
                        source="pasta sauce (estimated)",
                        estimated_calories=80,
                        confidence=0.25,
                        note="~0.5 cup tomato-based sauce assumed",
                    ))

        # ── Curry / stew over rice → check rice is accounted for
        if any(w in name for w in ["curry", "stew", "dal", "daal"]):
            all_names = " ".join(_get_name(i).lower() for i in items)
            if "rice" not in all_names and not has_rice_with_topping:
                has_rice_with_topping = True
                hidden.append(HiddenCalorieEstimate(
                    source="rice underneath (possibly hidden)",
                    estimated_calories=200,
                    confidence=0.2,
                    note="~1 cup cooked rice; confirm if no rice was served",
                ))

    # ── User description overrides
    # If the user explicitly says "no oil" / "no butter" / "no dressing",
    # remove those hidden estimates
    if desc_lower:
        negations = {
            "no oil": "cooking oil",
            "no butter": "butter",
            "no dressing": "dressing",
            "no sauce": "sauce",
            "no rice": "rice",
            "dry": "butter",
        }
        for phrase, source_keyword in negations.items():
            if phrase in desc_lower:
                hidden = [
                    h for h in hidden if source_keyword not in h.source.lower()
                ]

        # If user says "lots of oil" or "extra butter", boost those estimates
        if "lots of oil" in desc_lower or "extra oil" in desc_lower:
            for h in hidden:
                if "oil" in h.source.lower():
                    h.estimated_calories *= 1.5
                    h.note += " (user noted extra oil)"

    return hidden


# ── Helpers ──────────────────────────────────────────────────


def _get_name(item: FoodItem | FoodMatch) -> str:
    if hasattr(item, "name"):
        return item.name
    if hasattr(item, "item_name"):
        return item.item_name
    return ""


def _get_method(item: FoodItem | FoodMatch) -> str:
    if hasattr(item, "cooking_method"):
        return item.cooking_method
    return ""


def _get_additions(item: FoodItem | FoodMatch) -> list[str]:
    if hasattr(item, "visible_additions"):
        return item.visible_additions
    return []
