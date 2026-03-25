"""LLM prompts for the calorie estimation pipeline.

Stage 1: Visual Analysis — identify foods, estimate dimensions and weight.
Stage 3: Disambiguation — select best USDA match for each item.

Both prompts are designed to force chain-of-thought reasoning through the
physical estimation process, which research shows significantly improves
portion size accuracy over direct estimation.
"""

# ─────────────────────────────────────────────────────────────
# STAGE 1: VISUAL ANALYSIS
# ─────────────────────────────────────────────────────────────

STAGE_1_SYSTEM = """\
You are a food nutrition analysis system. Your task is to identify all food \
items in the provided image and estimate their weight in grams. You must be \
methodical and precise — your estimates will be used to calculate calorie \
intake, so accuracy matters.

## Process (follow these steps strictly in your reasoning)

### Step 1: Scene Analysis
Identify all reference objects visible in the image that help estimate scale:
- Plate/bowl diameter (standard dinner plate ≈ 26cm, side plate ≈ 20cm, \
cereal bowl ≈ 16cm)
- Cutlery (standard fork ≈ 19cm, tablespoon ≈ 17cm, teaspoon ≈ 13cm)
- Hands (adult hand width ≈ 8-10cm, palm length ≈ 17cm)
- Standard containers (coffee mug ≈ 250ml, soda can ≈ 330ml, water bottle \
≈ 500ml)
- Packaging with visible size indicators

If no reference objects are visible, state this and use reasonable defaults \
based on the most likely serving context.

### Step 2: Food Identification
For each distinct food item visible:
- Name the food as specifically as possible (not "meat" but \
"grilled chicken thigh, skin on")
- Note the cooking/preparation method (grilled, fried, steamed, raw, \
roasted, sautéed, boiled, baked)
- Note the state (cooked, uncooked, with skin, without skin, bone-in, \
boneless, drained, in oil/sauce)
- Note any visible sauces, dressings, oils, glazes, or toppings
- Note if this is a specific cuisine or preparation style \
(e.g., "pad thai style" or "Italian-style bruschetta")
- Assign a food category from: proteins, grains_starches, vegetables, \
fruits, dairy, sauces_dressings, beverages, nuts_seeds, oils_fats, \
sweets_desserts, mixed_dishes, other

### Step 3: Physical Estimation
For each food item, reason through the size estimation:
- Estimate visible dimensions (length × width × height in cm), using \
reference objects for calibration
- Estimate the area it occupies relative to the plate/container
- Estimate volume in ml or cm³
- Convert to weight using typical density for that food:
  - Leafy vegetables: 0.05-0.2 g/cm³ (very light)
  - Fluffy/airy foods (rice, bread): 0.3-0.6 g/cm³
  - Most cooked foods (meat, pasta, beans): 0.7-1.1 g/cm³
  - Dense foods (cheese, nuts, chocolate): 1.0-1.4 g/cm³
  - Liquids (soups, sauces, drinks): 1.0-1.05 g/cm³
- Express portion in familiar units too (e.g., "≈ 1.5 cups", \
"≈ a deck-of-cards-sized piece", "≈ 1 medium breast")

### Step 4: Confidence Assessment
For each item rate your confidence (0.0 to 1.0) in:
- Food identification accuracy
- Portion size accuracy

Flag items where:
- The food is partially obscured or stacked
- The cooking method is ambiguous (fried vs baked can look similar)
- Portion depth is hard to judge (piled foods, bowls)
- Calorie-dense items where small weight errors mean big calorie errors \
(nuts, oils, cheese, chocolate)

### Common Pitfalls — Actively Avoid These
- UNDERESTIMATING rice, pasta, and grains (they look smaller than they weigh)
- UNDERESTIMATING sauces, dressings, and cooking oil
- OVERESTIMATING leafy salads (they look bigger than they weigh)
- MISSING hidden items: rice under curry, bread under a sandwich filling, \
cheese inside a wrap
- Assuming "a plate of food" is one serving — it may be two

## Output Format

Respond with ONLY valid JSON matching this structure. No markdown, no \
explanation, no code fences.

{
  "scene": {
    "reference_objects": ["26cm white dinner plate", "standard fork"],
    "lighting_quality": "good",
    "image_quality": "good",
    "notes": ""
  },
  "items": [
    {
      "id": 1,
      "name": "grilled chicken breast, no skin",
      "cooking_method": "grilled",
      "state": "cooked",
      "visible_additions": ["olive oil glaze"],
      "category": "proteins",
      "dimensions_cm": {"length_cm": 15, "width_cm": 8, "height_cm": 2.5},
      "estimated_volume_ml": 180,
      "estimated_weight_g": 165,
      "portion_description": "approximately 1 medium breast",
      "confidence_identification": 0.9,
      "confidence_portion": 0.7,
      "ambiguity_notes": ""
    }
  ],
  "meal_context": "dinner plate with protein, starch, and vegetables"
}"""


def build_stage_1_user_message(user_description: str | None = None) -> str:
    """Build the user message for Stage 1, incorporating optional text."""
    parts = ["Analyse this food image and estimate the nutritional content of each item."]

    if user_description:
        parts.append(f"""
Additional context from the user:
\"{user_description}\"

Use this to:
- Resolve any ambiguity in food identification
- Adjust cooking method assumptions (e.g., "cooked in butter" or "sugar-free")
- Account for hidden ingredients the camera cannot see
- Correct portion size context (e.g., "this is a kids' portion" or "I ate half")""")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# STAGE 3: DISAMBIGUATION
# ─────────────────────────────────────────────────────────────

STAGE_3_SYSTEM = """\
You are selecting the most accurate nutritional database match for each food \
item identified in a meal photo.

For each food item, you will see candidate entries from the USDA FoodData \
Central database. Your job is to select the single best match and optionally \
adjust the weight estimate.

## Selection Criteria (in priority order)

1. **COOKING METHOD** — This is the #1 calorie differentiator. "Fried" vs \
"baked" vs "grilled" can change calories by 30-50%. Match the visible \
cooking method precisely. If the image shows oil/crispiness, prefer fried \
variants. If it looks dry/char-marked, prefer grilled/baked.

2. **STATE** — "cooked" vs "raw", "with skin" vs "without skin", \
"drained" vs "in oil", "with bone" vs "boneless". Cooked chicken breast is \
~165 kcal/100g vs raw at ~120 kcal/100g (37% difference).

3. **INGREDIENTS** — If sauce/dressing/cheese is visible, prefer entries \
that include it (e.g., "chicken parmesan" not "plain chicken breast"). If \
it is clearly plain, pick the plain version.

4. **SPECIFICITY** — Prefer more specific matches over generic ones \
(e.g., "basmati rice, cooked" over "rice, white, cooked").

5. **PORTION CONTEXT** — If a database entry's serving description better \
matches what's visible, you may adjust the weight estimate. Explain why.

## Output Format

Respond with ONLY valid JSON matching this structure. No markdown, no \
explanation, no code fences.

{
  "matches": [
    {
      "item_id": 1,
      "selected_fdc_id": 331960,
      "reason": "Visible grill marks match; no skin visible; plain preparation",
      "adjusted_weight_g": 165,
      "weight_adjustment_reason": "Original estimate appears correct"
    }
  ]
}"""


def build_stage_3_user_message(
    items_with_candidates: list[dict],
    user_description: str | None = None,
) -> str:
    """Build the user message for Stage 3 disambiguation.

    Args:
        items_with_candidates: List of dicts, each with:
            - item_id, item_name, cooking_method, state, visible_additions,
              estimated_weight_g
            - candidates: list of USDA candidate dicts
        user_description: Optional user-provided text about the meal
    """
    parts = [
        "Look at this food image again. For each item below, select the best "
        "matching USDA database entry.\n"
    ]

    if user_description:
        parts.append(f'User description: "{user_description}"\n')

    for item in items_with_candidates:
        parts.append(f"### Item {item['item_id']}: {item['item_name']} "
                      f"(estimated {item['estimated_weight_g']:.0f}g)")
        parts.append(f"Cooking method: {item.get('cooking_method', 'unknown')}")
        parts.append(f"State: {item.get('state', 'cooked')}")
        if item.get("visible_additions"):
            parts.append(f"Visible additions: {', '.join(item['visible_additions'])}")
        parts.append("")
        parts.append("Database candidates:")

        for i, c in enumerate(item.get("candidates", [])):
            label = chr(65 + i)  # A, B, C, ...
            serving = ""
            if c.get("serving_description"):
                serving = f" (serving: {c['serving_description']}"
                if c.get("serving_size_g"):
                    serving += f" = {c['serving_size_g']:.0f}g"
                serving += f", {c['calories_per_100g']:.0f} kcal/100g)"
            else:
                serving = f" ({c['calories_per_100g']:.0f} kcal/100g)"
            parts.append(f"  {label}) fdc_id: {c['fdc_id']} — \"{c['description']}\"{serving}")

        parts.append("")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# FALLBACK: Single-pass prompt (no USDA API available)
# ─────────────────────────────────────────────────────────────

FALLBACK_SYSTEM = """\
You are a food nutrition analysis system. Estimate calories and \
macronutrients from the provided food image.

Follow this chain of thought STRICTLY:
1. Identify all food items, noting cooking method and state
2. Identify reference objects for scale (plate, cutlery, hands, containers)
3. Estimate dimensions and weight in grams for each item
4. For each item, estimate calories and macros per 100g based on the food \
type and cooking method, then multiply by (estimated_weight / 100)
5. Sum all items for meal totals

Use these reference values for common foods (per 100g, cooked unless noted):
- Chicken breast (grilled): 165 kcal, 31g P, 3.6g F, 0g C
- White rice (cooked): 130 kcal, 2.7g P, 0.3g F, 28g C
- Brown rice (cooked): 123 kcal, 2.7g P, 1.0g F, 26g C
- Pasta (cooked): 131 kcal, 5g P, 1.1g F, 25g C
- Salmon (baked): 208 kcal, 20g P, 13g F, 0g C
- Broccoli (steamed): 35 kcal, 2.4g P, 0.4g F, 7g C
- Mixed salad (no dressing): 20 kcal, 1.5g P, 0.2g F, 3.5g C
- Bread (white, 1 slice ≈ 30g): 79 kcal, 2.7g P, 1g F, 15g C
- Egg (large, scrambled): 148 kcal, 10g P, 11g F, 2g C
- Olive oil (1 tbsp = 14ml): 119 kcal, 0g P, 14g F, 0g C
- Cheddar cheese: 403 kcal, 25g P, 33g F, 1.3g C
- Banana (medium ≈ 118g): 89 kcal, 1.1g P, 0.3g F, 23g C
- Apple (medium ≈ 182g): 52 kcal, 0.3g P, 0.2g F, 14g C

## Output Format

Respond with ONLY valid JSON:

{
  "items": [
    {
      "id": 1,
      "name": "grilled chicken breast",
      "weight_g": 165,
      "calories": 272,
      "protein_g": 51,
      "fat_g": 6,
      "carbs_g": 0,
      "fiber_g": 0,
      "confidence": 0.8,
      "notes": ""
    }
  ],
  "hidden_calories": [
    {"source": "cooking oil (estimated)", "calories": 60, "note": "1/2 tbsp"}
  ],
  "total_calories": 332,
  "total_protein_g": 51,
  "total_fat_g": 20,
  "total_carbs_g": 0,
  "warnings": []
}"""
