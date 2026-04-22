"""Data models for calorie estimation pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

ItemSource = Literal["usda", "barcode", "label_ocr", "fallback"]


class ImageQuality(str, Enum):
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


class FoodCategory(str, Enum):
    PROTEINS = "proteins"
    GRAINS_STARCHES = "grains_starches"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    DAIRY = "dairy"
    SAUCES_DRESSINGS = "sauces_dressings"
    BEVERAGES = "beverages"
    NUTS_SEEDS = "nuts_seeds"
    OILS_FATS = "oils_fats"
    SWEETS_DESSERTS = "sweets_desserts"
    MIXED_DISHES = "mixed_dishes"
    OTHER = "other"


# ── Stage 1 output: Visual Analysis ──────────────────────────


class SceneAnalysis(BaseModel):
    """What the model observes about the overall scene."""

    reference_objects: list[str] = Field(
        default_factory=list,
        description="Visible objects that help estimate scale (plate, fork, hand, etc.)",
    )
    lighting_quality: ImageQuality = ImageQuality.MODERATE
    image_quality: ImageQuality = ImageQuality.MODERATE
    notes: str = ""


class Dimensions(BaseModel):
    length_cm: float = 0
    width_cm: float = 0
    height_cm: float = 0


class FoodItem(BaseModel):
    """A single food item identified in the image."""

    id: int
    name: str = Field(description="Specific name, e.g. 'grilled chicken breast, no skin'")
    cooking_method: str = Field(default="", description="grilled, fried, steamed, raw, etc.")
    state: str = Field(default="cooked", description="cooked, raw, with skin, drained, etc.")
    visible_additions: list[str] = Field(
        default_factory=list,
        description="Sauces, dressings, toppings visible",
    )
    category: FoodCategory = FoodCategory.OTHER
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    estimated_volume_ml: float = 0
    estimated_weight_g: float = 0
    portion_description: str = Field(
        default="",
        description="Human-readable portion, e.g. '1 medium breast'",
    )
    confidence_identification: float = Field(ge=0, le=1, default=0.5)
    confidence_portion: float = Field(ge=0, le=1, default=0.5)
    ambiguity_notes: str = ""


class VisualAnalysis(BaseModel):
    """Complete output of Stage 1."""

    scene: SceneAnalysis = Field(default_factory=SceneAnalysis)
    items: list[FoodItem] = Field(default_factory=list)
    meal_context: str = ""


# ── Stage 2 output: USDA candidates ─────────────────────────


class USDACandidate(BaseModel):
    """A candidate food entry from the USDA database."""

    fdc_id: int
    description: str
    food_category: str = ""
    serving_size_g: float | None = None
    serving_description: str = ""
    calories_per_100g: float = 0
    protein_per_100g: float = 0
    fat_per_100g: float = 0
    carbs_per_100g: float = 0
    fiber_per_100g: float = 0
    sugar_per_100g: float = 0
    sodium_per_100g: float = 0
    saturated_fat_per_100g: float = 0


class ItemCandidates(BaseModel):
    """USDA candidates for a single food item."""

    item_id: int
    item_name: str
    candidates: list[USDACandidate] = Field(default_factory=list)


# ── Stage 0 alternate: Open Food Facts (barcode path) ───────


class OpenFoodFactsProduct(BaseModel):
    """A packaged-food product retrieved from the Open Food Facts API."""

    barcode: str
    product_name: str
    brand: str = ""
    serving_size_label: str = ""
    serving_quantity_g: float | None = None
    nutrients_per_100g: "NutrientProfile"
    data_quality_warnings: list[str] = Field(default_factory=list)

    def has_usable_nutrition(self) -> bool:
        """Guard against partial OFF records short-circuiting the pipeline.

        An OFF product is only authoritative when both the energy value and
        the main macros (protein / fat / carbs) are present. A hit with only
        ``energy-kcal_100g`` populated would otherwise render as a
        high-confidence packaged-food result with zero macros — worse than
        falling back to the vision pipeline or a label re-shoot.
        """
        n = self.nutrients_per_100g
        if n.calories <= 0:
            return False
        # For any real food with calories > 0, at least one of protein, fat,
        # or carbs must be non-zero by basic energy accounting. If all three
        # are zero the OFF record is missing macros rather than reporting
        # them as zero, so we decline the shortcut.
        return (n.protein_g + n.fat_g + n.carbs_g) > 0


class OFFContribution(BaseModel):
    """Payload extracted from a nutrition-label photo, ready to submit to OFF.

    Populated by the label-OCR path when the user re-shoots a packaged food
    whose barcode was detected but missing from Open Food Facts. The caller
    (bot / UI) is responsible for asking the user to confirm before submission.
    """

    barcode: str
    product_name: str
    brand: str = ""
    serving_size_label: str = ""
    serving_quantity_g: float | None = None
    nutrients_per_100g: "NutrientProfile"
    extraction_confidence: float = Field(ge=0, le=1, default=0.0)

    def is_submittable(self) -> bool:
        """Refuse to contribute partial/garbage data to OFF."""
        return (
            bool(self.barcode)
            and bool(self.product_name.strip())
            and self.nutrients_per_100g.calories > 0
        )


# ── Stage 3 output: Disambiguation ──────────────────────────


class FoodMatch(BaseModel):
    """Final matched food item after disambiguation."""

    item_id: int
    item_name: str
    selected_fdc_id: int
    selected_description: str
    reason: str = ""
    adjusted_weight_g: float
    weight_adjustment_reason: str = ""
    confidence_identification: float = 0.5
    confidence_portion: float = 0.5
    category: FoodCategory = FoodCategory.OTHER


# ── Stage 4 output: Final estimate ──────────────────────────


class NutrientProfile(BaseModel):
    """Nutritional values for a food item or meal total."""

    calories: float = 0
    protein_g: float = 0
    fat_g: float = 0
    carbs_g: float = 0
    fiber_g: float = 0
    sugar_g: float = 0
    sodium_mg: float = 0
    saturated_fat_g: float = 0

    def __add__(self, other: NutrientProfile) -> NutrientProfile:
        return NutrientProfile(
            calories=self.calories + other.calories,
            protein_g=self.protein_g + other.protein_g,
            fat_g=self.fat_g + other.fat_g,
            carbs_g=self.carbs_g + other.carbs_g,
            fiber_g=self.fiber_g + other.fiber_g,
            sugar_g=self.sugar_g + other.sugar_g,
            sodium_mg=self.sodium_mg + other.sodium_mg,
            saturated_fat_g=self.saturated_fat_g + other.saturated_fat_g,
        )

    def scale(self, factor: float) -> NutrientProfile:
        return NutrientProfile(
            calories=round(self.calories * factor, 1),
            protein_g=round(self.protein_g * factor, 1),
            fat_g=round(self.fat_g * factor, 1),
            carbs_g=round(self.carbs_g * factor, 1),
            fiber_g=round(self.fiber_g * factor, 1),
            sugar_g=round(self.sugar_g * factor, 1),
            sodium_mg=round(self.sodium_mg * factor, 1),
            saturated_fat_g=round(self.saturated_fat_g * factor, 1),
        )


class NutrientRange(BaseModel):
    """A range for low-confidence estimates."""

    low: NutrientProfile
    mid: NutrientProfile
    high: NutrientProfile


class ItemEstimate(BaseModel):
    """Final nutrition estimate for a single food item."""

    name: str
    weight_g: float
    nutrients: NutrientProfile
    range: NutrientRange | None = None
    fdc_id: int | None = None
    fdc_description: str = ""
    confidence: float = 0.5
    category: FoodCategory = FoodCategory.OTHER
    notes: str = ""
    source: ItemSource = "usda"
    # Populated for barcode and label_ocr sources so the user can see which
    # package serving was assumed and edit the amount if they ate more or less.
    barcode: str | None = None
    serving_size_label: str = ""


class HiddenCalorieEstimate(BaseModel):
    """Estimated hidden calories (cooking oil, etc.)."""

    source: str  # e.g. "cooking oil", "butter", "salad dressing"
    estimated_calories: float
    confidence: float = 0.3
    note: str = ""


class MealEstimate(BaseModel):
    """Complete meal estimation result."""

    items: list[ItemEstimate] = Field(default_factory=list)
    hidden_calories: list[HiddenCalorieEstimate] = Field(default_factory=list)
    total: NutrientProfile = Field(default_factory=NutrientProfile)
    total_with_hidden: NutrientProfile = Field(default_factory=NutrientProfile)
    overall_confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)
    meal_context: str = ""
    # Set when the barcode path failed (no data in OFF) so the caller can
    # prompt the user to re-send a nutrition-label photo on the next turn.
    needs_label_photo_for_barcode: str | None = None
    # Populated by the label_ocr path: nutrition extracted from a label photo,
    # ready to contribute back to Open Food Facts after user confirmation.
    pending_off_contribution: "OFFContribution | None" = None

    def format_summary(self, include_hidden: bool = True) -> str:
        """Format a human-readable summary for chat responses."""
        lines = ["🍽️ **Meal Estimate**\n"]

        for item in self.items:
            conf_dots = "●" * round(item.confidence * 5) + "○" * (5 - round(item.confidence * 5))
            lines.append(
                f"**{item.name}** ({item.weight_g:.0f}g) — "
                f"{item.nutrients.calories:.0f} kcal  "
                f"[{conf_dots}]"
            )
            lines.append(
                f"  P: {item.nutrients.protein_g:.0f}g · "
                f"F: {item.nutrients.fat_g:.0f}g · "
                f"C: {item.nutrients.carbs_g:.0f}g"
            )
            if item.source in ("barcode", "label_ocr"):
                serving = item.serving_size_label or f"{item.weight_g:.0f} g"
                provenance = (
                    "from package label"
                    if item.source == "barcode"
                    else "read from your label photo"
                )
                lines.append(
                    f"  📦 Per serving: {serving} "
                    f"({provenance} — adjust if you ate more or less)"
                )
            if item.range:
                lines.append(
                    f"  ↕ Range: {item.range.low.calories:.0f}–"
                    f"{item.range.high.calories:.0f} kcal"
                )
            lines.append("")

        if include_hidden and self.hidden_calories:
            lines.append("_Estimated hidden calories:_")
            for hc in self.hidden_calories:
                lines.append(f"  +{hc.estimated_calories:.0f} kcal — {hc.source} ({hc.note})")
            lines.append("")

        lines.append("─" * 36)

        use_total = self.total_with_hidden if include_hidden else self.total
        lines.append(f"**TOTAL: {use_total.calories:.0f} kcal**")
        lines.append(
            f"Protein: {use_total.protein_g:.0f}g · "
            f"Fat: {use_total.fat_g:.0f}g · "
            f"Carbs: {use_total.carbs_g:.0f}g · "
            f"Fiber: {use_total.fiber_g:.0f}g"
        )

        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"⚠️ {w}")

        if self.pending_off_contribution is not None:
            lines.append("")
            lines.append(
                "🆕 This product isn't in Open Food Facts yet. "
                "Reply *yes* to add it and help the next person."
            )

        return "\n".join(lines)

    def format_compact(self) -> str:
        """One-line summary for logging or inline display."""
        return (
            f"{self.total_with_hidden.calories:.0f} kcal | "
            f"P {self.total_with_hidden.protein_g:.0f}g "
            f"F {self.total_with_hidden.fat_g:.0f}g "
            f"C {self.total_with_hidden.carbs_g:.0f}g | "
            f"{len(self.items)} items"
        )


# Resolve forward references for models that reference each other by string.
OpenFoodFactsProduct.model_rebuild()
OFFContribution.model_rebuild()
MealEstimate.model_rebuild()
