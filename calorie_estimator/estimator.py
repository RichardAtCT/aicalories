"""Main CalorieEstimator orchestrator.

Runs the 4-stage pipeline:
  Stage 1: Visual analysis (LLM + vision)
  Stage 2: USDA database lookup
  Stage 3: Disambiguation (LLM, 2nd pass)
  Stage 4: Deterministic calculation

Supports Anthropic (default) and OpenAI as LLM providers.
Falls back to a single-pass estimation if the USDA API is unavailable.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from typing import Literal

from .corrections import (
    DEFAULT_BIAS_CORRECTIONS,
    apply_weight_correction,
    estimate_hidden_calories,
)
from .models import (
    FoodCategory,
    FoodMatch,
    HiddenCalorieEstimate,
    ItemEstimate,
    MealEstimate,
    NutrientProfile,
    NutrientRange,
    USDACandidate,
    VisualAnalysis,
)
from .prompts import (
    FALLBACK_SYSTEM,
    STAGE_1_SYSTEM,
    STAGE_3_SYSTEM,
    build_stage_1_user_message,
    build_stage_3_user_message,
)
from .usda import USDAClient

logger = logging.getLogger(__name__)


class CalorieEstimator:
    """Estimates calories and macronutrients from food photos.

    Usage:
        estimator = CalorieEstimator()
        result = await estimator.estimate(image_bytes, "grilled chicken with rice")
        print(result.format_summary())
    """

    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        usda_api_key: str | None = None,
        apply_bias_correction: bool = True,
        estimate_hidden_cals: bool = True,
        include_confidence_ranges: bool = True,
        temperature: float = 0.1,
        bias_corrections: dict[FoodCategory, float] | None = None,
    ):
        self.provider = provider
        self.temperature = temperature
        self.apply_bias_correction = apply_bias_correction
        self.estimate_hidden_cals = estimate_hidden_cals
        self.include_confidence_ranges = include_confidence_ranges
        self.bias_corrections = bias_corrections or DEFAULT_BIAS_CORRECTIONS

        # Set defaults per provider
        if provider == "anthropic":
            self.model = model or os.environ.get(
                "CALORIE_ESTIMATOR_MODEL", "claude-sonnet-4-20250514"
            )
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        elif provider == "openai":
            self.model = model or os.environ.get(
                "CALORIE_ESTIMATOR_MODEL", "gpt-4o"
            )
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.usda = USDAClient(api_key=usda_api_key)

    async def estimate(
        self,
        image: bytes,
        description: str | None = None,
        media_type: str | None = None,
    ) -> MealEstimate:
        """Run the full estimation pipeline.

        Args:
            image: Raw image bytes (JPEG, PNG, WebP, or GIF).
            description: Optional text description of the meal.
            media_type: MIME type of the image. Auto-detected if not provided.

        Returns:
            MealEstimate with per-item breakdown and totals.
        """
        if not media_type:
            media_type = _detect_media_type(image)

        image_b64 = base64.standard_b64encode(image).decode("utf-8")

        # ── Stage 1: Visual Analysis ─────────────────────────
        logger.info("Stage 1: Visual analysis")
        analysis = await self._stage_1_analyse(image_b64, media_type, description)

        if not analysis.items:
            return MealEstimate(
                warnings=["No food items detected in the image."],
                meal_context=analysis.meal_context,
            )

        # ── Stage 2: USDA Lookup ─────────────────────────────
        logger.info(f"Stage 2: USDA lookup for {len(analysis.items)} items")
        items_with_candidates = await self._stage_2_lookup(analysis)

        # Check if we got any candidates at all
        has_candidates = any(
            len(ic.get("candidates", [])) > 0
            for ic in items_with_candidates
        )

        if has_candidates:
            # ── Stage 3: Disambiguation ──────────────────────
            logger.info("Stage 3: Disambiguation")
            matches = await self._stage_3_disambiguate(
                image_b64, media_type, items_with_candidates, description
            )

            # ── Stage 4: Calculation ─────────────────────────
            logger.info("Stage 4: Calculation")
            return self._stage_4_calculate(
                matches, items_with_candidates, analysis, description
            )
        else:
            # Fallback: use Stage 1 estimates directly with fallback nutrition values
            logger.warning("No USDA candidates found; using fallback estimation")
            return await self._fallback_estimate(image_b64, media_type, description)

    # ─────────────────────────────────────────────────────────
    # Stage 1: Visual Analysis
    # ─────────────────────────────────────────────────────────

    async def _stage_1_analyse(
        self,
        image_b64: str,
        media_type: str,
        description: str | None,
    ) -> VisualAnalysis:
        """Call the LLM with the image to identify and measure food items."""
        user_text = build_stage_1_user_message(description)

        response_text = await self._call_llm(
            system=STAGE_1_SYSTEM,
            user_text=user_text,
            image_b64=image_b64,
            media_type=media_type,
        )

        try:
            data = _parse_json(response_text)
            return VisualAnalysis.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse Stage 1 output: {e}")
            logger.debug(f"Raw output: {response_text[:500]}")
            return VisualAnalysis()

    # ─────────────────────────────────────────────────────────
    # Stage 2: USDA Lookup
    # ─────────────────────────────────────────────────────────

    async def _stage_2_lookup(
        self,
        analysis: VisualAnalysis,
    ) -> list[dict]:
        """For each food item, search USDA for candidate matches."""
        results = []
        for item in analysis.items:
            # Build a specific search query
            query_parts = []
            if item.cooking_method:
                query_parts.append(item.cooking_method)
            query_parts.append(item.name)
            if item.state and item.state != "cooked":
                query_parts.append(item.state)
            query = " ".join(query_parts)

            candidates = await self.usda.search(query, max_results=5)

            # If no results, try a simpler query (just the name)
            if not candidates:
                candidates = await self.usda.search(item.name, max_results=5)

            results.append({
                "item_id": item.id,
                "item_name": item.name,
                "cooking_method": item.cooking_method,
                "state": item.state,
                "visible_additions": item.visible_additions,
                "estimated_weight_g": item.estimated_weight_g,
                "category": item.category,
                "confidence_identification": item.confidence_identification,
                "confidence_portion": item.confidence_portion,
                "candidates": [c.model_dump() for c in candidates],
            })

        return results

    # ─────────────────────────────────────────────────────────
    # Stage 3: Disambiguation
    # ─────────────────────────────────────────────────────────

    async def _stage_3_disambiguate(
        self,
        image_b64: str,
        media_type: str,
        items_with_candidates: list[dict],
        description: str | None,
    ) -> list[FoodMatch]:
        """Second LLM call to select best USDA match per item."""
        user_text = build_stage_3_user_message(items_with_candidates, description)

        response_text = await self._call_llm(
            system=STAGE_3_SYSTEM,
            user_text=user_text,
            image_b64=image_b64,
            media_type=media_type,
        )

        try:
            data = _parse_json(response_text)
            matches_raw = data.get("matches", [])
        except Exception as e:
            logger.error(f"Failed to parse Stage 3 output: {e}")
            matches_raw = []

        # Build FoodMatch objects, merging in confidence and category from Stage 1
        item_lookup = {ic["item_id"]: ic for ic in items_with_candidates}
        matches = []
        for m in matches_raw:
            item_id = m.get("item_id")
            ic = item_lookup.get(item_id, {})
            # Find the selected candidate's description
            selected_desc = ""
            for c in ic.get("candidates", []):
                if c["fdc_id"] == m.get("selected_fdc_id"):
                    selected_desc = c["description"]
                    break

            matches.append(FoodMatch(
                item_id=item_id,
                item_name=ic.get("item_name", ""),
                selected_fdc_id=m.get("selected_fdc_id", 0),
                selected_description=selected_desc,
                reason=m.get("reason", ""),
                adjusted_weight_g=m.get("adjusted_weight_g", ic.get("estimated_weight_g", 0)),
                weight_adjustment_reason=m.get("weight_adjustment_reason", ""),
                confidence_identification=ic.get("confidence_identification", 0.5),
                confidence_portion=ic.get("confidence_portion", 0.5),
                category=FoodCategory(ic.get("category", "other")),
            ))

        return matches

    # ─────────────────────────────────────────────────────────
    # Stage 4: Deterministic Calculation
    # ─────────────────────────────────────────────────────────

    def _stage_4_calculate(
        self,
        matches: list[FoodMatch],
        items_with_candidates: list[dict],
        analysis: VisualAnalysis,
        description: str | None,
    ) -> MealEstimate:
        """Calculate final nutrition values from matches + DB data."""
        # Build lookup of candidates by fdc_id for nutrient data
        candidate_lookup: dict[int, USDACandidate] = {}
        for ic in items_with_candidates:
            for c in ic["candidates"]:
                candidate_lookup[c["fdc_id"]] = USDACandidate(**c)

        items: list[ItemEstimate] = []
        meal_total = NutrientProfile()

        for match in matches:
            db_entry = candidate_lookup.get(match.selected_fdc_id)
            if not db_entry:
                logger.warning(f"No DB entry for fdc_id {match.selected_fdc_id}")
                continue

            weight_g = match.adjusted_weight_g

            # Apply bias correction
            if self.apply_bias_correction:
                weight_g = apply_weight_correction(
                    weight_g, match.category, self.bias_corrections
                )

            # Calculate nutrients: DB values are per 100g
            scale = weight_g / 100.0
            nutrients = NutrientProfile(
                calories=round(db_entry.calories_per_100g * scale, 1),
                protein_g=round(db_entry.protein_per_100g * scale, 1),
                fat_g=round(db_entry.fat_per_100g * scale, 1),
                carbs_g=round(db_entry.carbs_per_100g * scale, 1),
                fiber_g=round(db_entry.fiber_per_100g * scale, 1),
                sugar_g=round(db_entry.sugar_per_100g * scale, 1),
                sodium_mg=round(db_entry.sodium_per_100g * scale, 1),
                saturated_fat_g=round(db_entry.saturated_fat_per_100g * scale, 1),
            )

            # Confidence-weighted range
            confidence = min(match.confidence_identification, match.confidence_portion)
            nutrient_range = None
            if self.include_confidence_ranges and confidence < 0.7:
                margin = 0.25 if confidence < 0.5 else 0.15
                nutrient_range = NutrientRange(
                    low=nutrients.scale(1 - margin),
                    mid=nutrients,
                    high=nutrients.scale(1 + margin),
                )

            items.append(ItemEstimate(
                name=match.item_name,
                weight_g=round(weight_g, 0),
                nutrients=nutrients,
                range=nutrient_range,
                fdc_id=match.selected_fdc_id,
                fdc_description=match.selected_description,
                confidence=confidence,
                category=match.category,
            ))

            meal_total = meal_total + nutrients

        # Hidden calories
        hidden: list[HiddenCalorieEstimate] = []
        if self.estimate_hidden_cals:
            hidden = estimate_hidden_calories(analysis.items, description)

        total_with_hidden = NutrientProfile(
            calories=meal_total.calories + sum(h.estimated_calories for h in hidden),
            protein_g=meal_total.protein_g,
            fat_g=meal_total.fat_g + sum(
                h.estimated_calories / 9 for h in hidden  # rough: assume hidden cals are fat
            ),
            carbs_g=meal_total.carbs_g,
            fiber_g=meal_total.fiber_g,
            sugar_g=meal_total.sugar_g,
            sodium_mg=meal_total.sodium_mg,
            saturated_fat_g=meal_total.saturated_fat_g,
        )

        # Warnings
        warnings = []
        low_conf_items = [i for i in items if i.confidence < 0.5]
        if low_conf_items:
            names = ", ".join(i.name for i in low_conf_items)
            warnings.append(f"Low confidence for: {names}. Consider adding a text description.")
        if analysis.scene.image_quality == "poor":
            warnings.append("Image quality is poor — estimates may be less accurate.")
        if hidden:
            warnings.append(
                "Hidden calorie estimates included (cooking oil, etc.). "
                "Add a description to refine these."
            )

        overall_confidence = (
            sum(i.confidence for i in items) / len(items) if items else 0
        )

        return MealEstimate(
            items=items,
            hidden_calories=hidden,
            total=meal_total,
            total_with_hidden=total_with_hidden,
            overall_confidence=round(overall_confidence, 2),
            warnings=warnings,
            meal_context=analysis.meal_context,
        )

    # ─────────────────────────────────────────────────────────
    # Fallback: Single-pass (no USDA data)
    # ─────────────────────────────────────────────────────────

    async def _fallback_estimate(
        self,
        image_b64: str,
        media_type: str,
        description: str | None,
    ) -> MealEstimate:
        """Single-pass estimation when USDA data is unavailable."""
        user_text = "Estimate the calories and macronutrients in this food image."
        if description:
            user_text += f'\n\nUser description: "{description}"'

        response_text = await self._call_llm(
            system=FALLBACK_SYSTEM,
            user_text=user_text,
            image_b64=image_b64,
            media_type=media_type,
        )

        try:
            data = _parse_json(response_text)
        except Exception as e:
            logger.error(f"Failed to parse fallback output: {e}")
            return MealEstimate(warnings=["Failed to analyse the image."])

        items = []
        for raw_item in data.get("items", []):
            nutrients = NutrientProfile(
                calories=raw_item.get("calories", 0),
                protein_g=raw_item.get("protein_g", 0),
                fat_g=raw_item.get("fat_g", 0),
                carbs_g=raw_item.get("carbs_g", 0),
                fiber_g=raw_item.get("fiber_g", 0),
            )
            items.append(ItemEstimate(
                name=raw_item.get("name", "Unknown"),
                weight_g=raw_item.get("weight_g", 0),
                nutrients=nutrients,
                confidence=raw_item.get("confidence", 0.4),
                notes=raw_item.get("notes", ""),
            ))

        meal_total = NutrientProfile()
        for item in items:
            meal_total = meal_total + item.nutrients

        hidden = [
            HiddenCalorieEstimate(
                source=h["source"],
                estimated_calories=h["calories"],
                note=h.get("note", ""),
            )
            for h in data.get("hidden_calories", [])
        ]

        total_with_hidden = NutrientProfile(
            calories=meal_total.calories + sum(h.estimated_calories for h in hidden),
            protein_g=meal_total.protein_g,
            fat_g=meal_total.fat_g,
            carbs_g=meal_total.carbs_g,
            fiber_g=meal_total.fiber_g,
        )

        warnings = data.get("warnings", [])
        warnings.append("⚠️ USDA database unavailable — using LLM internal estimates (less accurate).")

        return MealEstimate(
            items=items,
            hidden_calories=hidden,
            total=meal_total,
            total_with_hidden=total_with_hidden,
            overall_confidence=0.4,
            warnings=warnings,
        )

    # ─────────────────────────────────────────────────────────
    # LLM Calling (Anthropic / OpenAI)
    # ─────────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system: str,
        user_text: str,
        image_b64: str,
        media_type: str,
    ) -> str:
        """Call the vision LLM and return the text response."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system, user_text, image_b64, media_type)
        elif self.provider == "openai":
            return await self._call_openai(system, user_text, image_b64, media_type)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _call_anthropic(
        self,
        system: str,
        user_text: str,
        image_b64: str,
        media_type: str,
    ) -> str:
        """Call Anthropic's Messages API with vision."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        message = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                }
            ],
        )

        return message.content[0].text

    async def _call_openai(
        self,
        system: str,
        user_text: str,
        image_b64: str,
        media_type: str,
    ) -> str:
        """Call OpenAI's Chat Completions API with vision."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        )

        return response.choices[0].message.content

    # ─────────────────────────────────────────────────────────
    # Agent Tool Interface
    # ─────────────────────────────────────────────────────────

    async def estimate_from_base64(
        self,
        image_base64: str,
        description: str | None = None,
        media_type: str = "image/jpeg",
    ) -> MealEstimate:
        """Convenience method for agent tool integrations.

        Accepts base64 string directly (no raw bytes needed).
        """
        image_bytes = base64.standard_b64decode(image_base64)
        return await self.estimate(image_bytes, description, media_type)

    def get_tool_definition(self) -> dict:
        """Return a tool definition dict for agent frameworks.

        Compatible with Anthropic tool_use, OpenAI function calling,
        LangChain tools, etc.
        """
        return {
            "name": "estimate_calories",
            "description": (
                "Estimate calories and macronutrients from a food photo. "
                "Returns a detailed per-item breakdown with confidence scores, "
                "plus hidden calorie estimates for cooking oil, sauces, etc. "
                "Provide a text description for better accuracy (cooking method, "
                "hidden ingredients, portion context)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded food image (JPEG or PNG)",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Optional text description: cooking method, "
                            "hidden ingredients, portion context"
                        ),
                    },
                    "media_type": {
                        "type": "string",
                        "enum": ["image/jpeg", "image/png", "image/webp", "image/gif"],
                        "description": "MIME type of the image (default: image/jpeg)",
                    },
                },
                "required": ["image_base64"],
            },
        }


# ── Helpers ──────────────────────────────────────────────────


def _detect_media_type(image_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"  # default fallback


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown code fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)
