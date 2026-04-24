"""Main CalorieEstimator orchestrator.

Runs the 4-stage pipeline:
  Stage 1: Visual analysis (LLM + vision)
  Stage 2: USDA database lookup
  Stage 3: Disambiguation (LLM, 2nd pass)
  Stage 4: Deterministic calculation

Supports OpenAI Codex (default), Anthropic, and OpenAI as LLM providers.
Falls back to a single-pass estimation if the USDA API is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Literal

from . import barcode as barcode_detector
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
    OFFContribution,
    OpenFoodFactsProduct,
    USDACandidate,
    VisualAnalysis,
)
from .openfoodfacts import OpenFoodFactsClient
from .prompts import (
    FALLBACK_SYSTEM,
    LABEL_OCR_SYSTEM,
    STAGE_1_SYSTEM,
    STAGE_3_SYSTEM,
    TEXT_EXTRACTION_SYSTEM,
    build_label_ocr_user_message,
    build_stage_1_user_message,
    build_stage_3_user_message,
    build_text_extraction_user_message,
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
        provider: Literal["anthropic", "openai", "openai-codex"] = "openai-codex",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
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
        self.base_url = base_url or os.environ.get("CALORIE_ESTIMATOR_BASE_URL")

        # Set defaults per provider
        if provider == "anthropic":
            self.model = model or os.environ.get(
                "CALORIE_ESTIMATOR_MODEL", "claude-sonnet-4-20250514"
            )
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        elif provider == "openai":
            self.model = model or os.environ.get("CALORIE_ESTIMATOR_MODEL", "gpt-4o")
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        elif provider == "openai-codex":
            self.model = model or os.environ.get("CALORIE_ESTIMATOR_MODEL", "gpt-5.4")
            self.base_url = (
                base_url
                or os.environ.get("CALORIE_ESTIMATOR_BASE_URL")
                or "https://chatgpt.com/backend-api/codex"
            )
            self.api_key = (
                api_key
                or os.environ.get("CODEX_ACCESS_TOKEN", "")
                or self._read_codex_access_token()
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.usda = USDAClient(api_key=usda_api_key)
        self.off = OpenFoodFactsClient()

    async def estimate(
        self,
        image: bytes,
        description: str | None = None,
        media_type: str | None = None,
        barcode_hint: str | None = None,
    ) -> MealEstimate:
        """Run the full estimation pipeline.

        Args:
            image: Raw image bytes (JPEG, PNG, WebP, or GIF).
            description: Optional text description of the meal.
            media_type: MIME type of the image. Auto-detected if not provided.
            barcode_hint: If set, treat this image as a nutrition-label
                re-shoot for the given UPC/EAN — run the label-OCR path and
                attach a pending Open Food Facts contribution to the result.
                Callers typically set this after a previous turn returned a
                MealEstimate with ``needs_label_photo_for_barcode`` populated.

        Returns:
            MealEstimate with per-item breakdown and totals.
        """
        if not media_type:
            media_type = _detect_media_type(image)

        image_b64 = base64.standard_b64encode(image).decode("utf-8")

        # ── Label-OCR path: caller told us this is a re-shot label ───
        if barcode_hint:
            logger.info("Label-OCR path for barcode %s", barcode_hint)
            return await self._stage_label_ocr(image_b64, media_type, barcode_hint)

        # ── Stage 0: Barcode → Open Food Facts ──────────────
        stage_0_result = await self._stage_0_barcode(image)
        if stage_0_result is not None:
            return stage_0_result

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
            len(ic.get("candidates", [])) > 0 for ic in items_with_candidates
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

    async def estimate_from_text(self, description: str) -> MealEstimate:
        """Run the pipeline from a plain-text description (no image).

        A text-only Stage 1 LLM call extracts structured ``FoodItem`` entries
        from the description, then the existing Stage 2 (USDA lookup), Stage 3
        (disambiguation, no image), and Stage 4 (deterministic calculation)
        run unchanged.

        Args:
            description: Natural-language description of the meal,
                e.g. "two slices of pepperoni pizza".

        Returns:
            MealEstimate with per-item breakdown and totals. Empty with a
            warning when the description is too vague or has no USDA matches.
        """
        logger.info("Stage 1 (text): extraction")
        analysis = await self._stage_1_extract_from_text(description)

        if not analysis.items:
            return MealEstimate(
                warnings=[
                    "Couldn't pick out any food items from the description. "
                    "Try being more specific about what you ate."
                ],
            )

        logger.info(f"Stage 2: USDA lookup for {len(analysis.items)} items")
        items_with_candidates = await self._stage_2_lookup(analysis)

        if not any(ic.get("candidates") for ic in items_with_candidates):
            return MealEstimate(
                warnings=[
                    "No matching nutrition data found for the items described. "
                    "Try more common food names."
                ],
                meal_context=analysis.meal_context,
            )

        logger.info("Stage 3 (text): disambiguation")
        matches = await self._stage_3_disambiguate(
            None, None, items_with_candidates, description
        )

        logger.info("Stage 4: calculation")
        return self._stage_4_calculate(
            matches, items_with_candidates, analysis, description
        )

    async def _stage_1_extract_from_text(self, description: str) -> VisualAnalysis:
        """Text-only Stage 1: extract structured FoodItems from a description."""
        response_text = await self._call_llm(
            system=TEXT_EXTRACTION_SYSTEM,
            user_text=build_text_extraction_user_message(description),
        )

        try:
            data = _parse_json(response_text)
            data = _normalize_enums(data)
            return VisualAnalysis.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse text-extraction output: {e}")
            logger.debug(f"Raw output: {response_text[:500]}")
            return VisualAnalysis()

    # ─────────────────────────────────────────────────────────
    # Stage 0: Barcode → Open Food Facts
    # ─────────────────────────────────────────────────────────

    async def _stage_0_barcode(self, image: bytes) -> MealEstimate | None:
        """Detect barcodes and resolve the first usable one via OFF.

        Returns:
            - A complete ``MealEstimate`` if a barcode resolves to an OFF
              product with usable nutrition.
            - An empty ``MealEstimate`` with a warning + ``needs_label_photo_for_barcode``
              set if barcodes are found but OFF has no usable data for any of them.
            - ``None`` when no barcode is detected, so the caller can fall
              through to the regular visual pipeline.
        """
        codes = barcode_detector.detect_barcodes(image)
        if not codes:
            return None

        logger.info("Detected barcodes: %s", codes)

        resolved_but_empty: list[str] = []
        for code in codes:
            product = await self.off.lookup(code)
            if product is None:
                continue
            if product.has_usable_nutrition():
                return _meal_from_off_product(product)
            resolved_but_empty.append(code)

        # We saw barcodes — surface to the caller so they can prompt a
        # nutrition-label re-shoot. The first detected code wins as the
        # "hint" the caller should feed back in on the next turn.
        if resolved_but_empty:
            warning = (
                f"Barcode {resolved_but_empty[0]} was found in Open Food Facts "
                "but has no nutrition data. Please send a photo of the nutrition "
                "label, or of the food itself, so I can estimate it."
            )
        else:
            warning = (
                f"Barcode {codes[0]} isn't in Open Food Facts yet. Please send a "
                "photo of the nutrition label (and I'll add it for others), or "
                "of the food itself if there is no label."
            )
        return MealEstimate(
            warnings=[warning],
            needs_label_photo_for_barcode=codes[0],
        )

    # ─────────────────────────────────────────────────────────
    # Label OCR: read a re-shot nutrition label
    # ─────────────────────────────────────────────────────────

    async def _stage_label_ocr(
        self,
        image_b64: str,
        media_type: str,
        barcode: str,
    ) -> MealEstimate:
        """Read a nutrition label, return a MealEstimate + pending OFF contribution."""
        response_text = await self._call_llm(
            system=LABEL_OCR_SYSTEM,
            user_text=build_label_ocr_user_message(barcode),
            image_b64=image_b64,
            media_type=media_type,
        )

        try:
            data = _parse_json(response_text)
        except Exception as e:
            logger.error(f"Failed to parse label-OCR output: {e}")
            return MealEstimate(
                warnings=[
                    "Couldn't read the nutrition label. Please try again with a "
                    "closer, well-lit photo of the label — or a photo of the food "
                    "itself and I'll estimate it visually."
                ],
                needs_label_photo_for_barcode=barcode,
            )

        product_name = (data.get("product_name") or "").strip()
        nutrients_raw = data.get("nutrients_per_100g") or {}
        extraction_conf = float(data.get("extraction_confidence") or 0.0)

        nutrients_per_100g = NutrientProfile(
            calories=_safe_float(nutrients_raw.get("calories")),
            protein_g=_safe_float(nutrients_raw.get("protein_g")),
            fat_g=_safe_float(nutrients_raw.get("fat_g")),
            carbs_g=_safe_float(nutrients_raw.get("carbs_g")),
            fiber_g=_safe_float(nutrients_raw.get("fiber_g")),
            sugar_g=_safe_float(nutrients_raw.get("sugar_g")),
            sodium_mg=_safe_float(nutrients_raw.get("sodium_mg")),
            saturated_fat_g=_safe_float(nutrients_raw.get("saturated_fat_g")),
        )

        if not product_name or nutrients_per_100g.calories <= 0:
            return MealEstimate(
                warnings=[
                    "I couldn't read the calorie information from that label. "
                    "Try a closer, straight-on photo — or send a photo of the "
                    "food itself and I'll estimate it visually."
                ],
                needs_label_photo_for_barcode=barcode,
            )

        serving_size_label = (data.get("serving_size_label") or "").strip()
        serving_quantity_g = _safe_float(data.get("serving_quantity_g")) or None

        # Build a single ItemEstimate for the serving. Prefer the label's
        # declared serving weight; fall back to 100 g with a warning.
        warnings: list[str] = []
        if serving_quantity_g and serving_quantity_g > 0:
            weight_g = serving_quantity_g
            scale = weight_g / 100.0
            serving_display = serving_size_label or f"{weight_g:.0f} g"
        else:
            weight_g = 100.0
            scale = 1.0
            serving_display = serving_size_label or "100 g"
            warnings.append(
                "Couldn't read the serving size from the label — showing "
                "nutrition per 100 g. Adjust the amount if you ate a different "
                "quantity."
            )

        nutrients_serving = nutrients_per_100g.scale(scale)

        item = ItemEstimate(
            name=product_name,
            weight_g=round(weight_g, 0),
            nutrients=nutrients_serving,
            confidence=max(0.0, min(1.0, extraction_conf or 0.75)),
            source="label_ocr",
            barcode=barcode,
            serving_size_label=serving_display,
            notes=(data.get("notes") or "").strip(),
        )

        contribution = OFFContribution(
            barcode=barcode,
            product_name=product_name,
            brand=(data.get("brand") or "").strip(),
            serving_size_label=serving_size_label,
            serving_quantity_g=serving_quantity_g,
            nutrients_per_100g=nutrients_per_100g,
            extraction_confidence=extraction_conf,
        )

        return MealEstimate(
            items=[item],
            total=nutrients_serving,
            total_with_hidden=nutrients_serving,
            overall_confidence=item.confidence,
            warnings=warnings,
            pending_off_contribution=(
                contribution if contribution.is_submittable() else None
            ),
        )

    async def submit_pending_contribution(self, contribution: OFFContribution) -> bool:
        """Submit a user-confirmed contribution to Open Food Facts.

        Thin wrapper around ``OpenFoodFactsClient.submit`` so callers don't
        need to reach into ``self.off`` directly.
        """
        return await self.off.submit(contribution)

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
            data = _normalize_enums(data)
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

            results.append(
                {
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
                }
            )

        return results

    # ─────────────────────────────────────────────────────────
    # Stage 3: Disambiguation
    # ─────────────────────────────────────────────────────────

    async def _stage_3_disambiguate(
        self,
        image_b64: str | None,
        media_type: str | None,
        items_with_candidates: list[dict],
        description: str | None,
    ) -> list[FoodMatch]:
        """Second LLM call to select best USDA match per item.

        ``image_b64`` / ``media_type`` may be None when called from the
        text-only path (``estimate_from_text``); the LLM disambiguates from
        the candidate descriptions alone in that case.
        """
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

            matches.append(
                FoodMatch(
                    item_id=item_id,
                    item_name=ic.get("item_name", ""),
                    selected_fdc_id=m.get("selected_fdc_id", 0),
                    selected_description=selected_desc,
                    reason=m.get("reason", ""),
                    adjusted_weight_g=m.get(
                        "adjusted_weight_g", ic.get("estimated_weight_g", 0)
                    ),
                    weight_adjustment_reason=m.get("weight_adjustment_reason", ""),
                    confidence_identification=ic.get("confidence_identification", 0.5),
                    confidence_portion=ic.get("confidence_portion", 0.5),
                    category=FoodCategory(ic.get("category", "other")),
                )
            )

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

            items.append(
                ItemEstimate(
                    name=match.item_name,
                    weight_g=round(weight_g, 0),
                    nutrients=nutrients,
                    range=nutrient_range,
                    fdc_id=match.selected_fdc_id,
                    fdc_description=match.selected_description,
                    confidence=confidence,
                    category=match.category,
                )
            )

            meal_total = meal_total + nutrients

        # Hidden calories
        hidden: list[HiddenCalorieEstimate] = []
        if self.estimate_hidden_cals:
            hidden = estimate_hidden_calories(analysis.items, description)

        total_with_hidden = NutrientProfile(
            calories=meal_total.calories + sum(h.estimated_calories for h in hidden),
            protein_g=meal_total.protein_g,
            fat_g=meal_total.fat_g
            + sum(
                h.estimated_calories / 9
                for h in hidden  # rough: assume hidden cals are fat
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
            warnings.append(
                f"Low confidence for: {names}. Consider adding a text description."
            )
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
            items.append(
                ItemEstimate(
                    name=raw_item.get("name", "Unknown"),
                    weight_g=raw_item.get("weight_g", 0),
                    nutrients=nutrients,
                    confidence=raw_item.get("confidence", 0.4),
                    notes=raw_item.get("notes", ""),
                )
            )

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
        warnings.append(
            "⚠️ USDA database unavailable — using LLM internal estimates (less accurate)."
        )

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
        image_b64: str | None = None,
        media_type: str | None = None,
    ) -> str:
        """Call the LLM and return the text response. Image is optional."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system, user_text, image_b64, media_type)
        elif self.provider == "openai":
            return await self._call_openai(system, user_text, image_b64, media_type)
        elif self.provider == "openai-codex":
            return await self._call_openai_codex(
                system, user_text, image_b64, media_type
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _read_codex_access_token(self) -> str:
        """Read a Codex OAuth token from env or Hermes auth.json when available."""
        env_token = os.environ.get("CODEX_ACCESS_TOKEN", "").strip()
        if env_token:
            return env_token

        hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        auth_path = hermes_home / "auth.json"
        try:
            data = json.loads(auth_path.read_text())
            return str(
                data.get("providers", {})
                .get("openai-codex", {})
                .get("tokens", {})
                .get("access_token", "")
            ).strip()
        except Exception:
            return ""

    @staticmethod
    def codex_auth_available() -> bool:
        """Return True when a Codex token is actually available."""
        env_token = os.environ.get("CODEX_ACCESS_TOKEN", "").strip()
        if env_token:
            return True

        hermes_home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
        auth_path = hermes_home / "auth.json"
        try:
            data = json.loads(auth_path.read_text())
            token = str(
                data.get("providers", {})
                .get("openai-codex", {})
                .get("tokens", {})
                .get("access_token", "")
            ).strip()
            return bool(token)
        except Exception:
            return False

    def _codex_cloudflare_headers(self, access_token: str) -> dict[str, str]:
        """Headers required for ChatGPT Codex endpoint access."""
        headers = {
            "User-Agent": "codex_cli_rs/0.0.0 (AI Calories)",
            "originator": "codex_cli_rs",
        }
        if not access_token:
            return headers
        try:
            parts = access_token.split(".")
            if len(parts) >= 2:
                payload = parts[1] + "=" * (-len(parts[1]) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload))
                acct_id = claims.get("https://api.openai.com/auth", {}).get(
                    "chatgpt_account_id"
                )
                if isinstance(acct_id, str) and acct_id:
                    headers["ChatGPT-Account-ID"] = acct_id
        except Exception:
            pass
        return headers

    async def _call_anthropic(
        self,
        system: str,
        user_text: str,
        image_b64: str | None,
        media_type: str | None,
    ) -> str:
        """Call Anthropic's Messages API. Image is optional."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        content: list[dict] = []
        if image_b64 and media_type:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        message = await client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

        return message.content[0].text

    async def _call_openai(
        self,
        system: str,
        user_text: str,
        image_b64: str | None,
        media_type: str | None,
    ) -> str:
        """Call OpenAI's Chat Completions API. Image is optional."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            **({"base_url": self.base_url} if self.base_url else {}),
        )

        user_content: list[dict] = []
        if image_b64 and media_type:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                        "detail": "high",
                    },
                }
            )
        user_content.append({"type": "text", "text": user_text})

        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )

        return response.choices[0].message.content

    async def _call_openai_codex(
        self,
        system: str,
        user_text: str,
        image_b64: str | None,
        media_type: str | None,
    ) -> str:
        """Call ChatGPT Codex Responses API without blocking the event loop. Image is optional."""
        return await asyncio.to_thread(
            self._call_openai_codex_sync,
            system,
            user_text,
            image_b64,
            media_type,
        )

    def _call_openai_codex_sync(
        self,
        system: str,
        user_text: str,
        image_b64: str | None,
        media_type: str | None,
    ) -> str:
        """Synchronous Codex call, executed in a worker thread by async callers."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self._codex_cloudflare_headers(self.api_key),
        )

        user_content: list[dict] = []
        if image_b64 and media_type:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{image_b64}",
                    "detail": "high",
                }
            )
        user_content.append({"type": "input_text", "text": user_text})

        with client.responses.stream(
            model=self.model,
            store=False,
            instructions=system,
            input=[{"role": "user", "content": user_content}],
        ) as stream:
            text_parts: list[str] = []
            for event in stream:
                etype = getattr(event, "type", "")
                if "output_text.delta" in etype:
                    delta = getattr(event, "delta", "")
                    if delta:
                        text_parts.append(delta)
            final = stream.get_final_response()

        return "".join(text_parts).strip() or getattr(final, "output_text", "") or ""

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


def _safe_float(value) -> float:
    """Coerce JSON values to a float, treating ``None`` / strings as 0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _meal_from_off_product(product: OpenFoodFactsProduct) -> MealEstimate:
    """Build a complete MealEstimate from an Open Food Facts product hit.

    Uses the package serving when OFF provides one, otherwise falls back
    to per-100 g with a warning so the user knows to adjust.
    """
    warnings: list[str] = []
    if product.serving_quantity_g and product.serving_quantity_g > 0:
        weight_g = product.serving_quantity_g
        scale = weight_g / 100.0
        serving_display = product.serving_size_label or f"{weight_g:.0f} g"
    else:
        weight_g = 100.0
        scale = 1.0
        serving_display = product.serving_size_label or "100 g"
        warnings.append(
            "No serving size on this product in Open Food Facts — showing "
            "nutrition per 100 g. Adjust the amount if you ate a different "
            "quantity."
        )

    nutrients = product.nutrients_per_100g.scale(scale)

    display_name = product.product_name
    if product.brand:
        display_name = f"{product.brand} — {product.product_name}"

    item = ItemEstimate(
        name=display_name,
        weight_g=round(weight_g, 0),
        nutrients=nutrients,
        confidence=0.9,
        source="barcode",
        barcode=product.barcode,
        serving_size_label=serving_display,
    )

    if product.data_quality_warnings:
        warnings.append(
            "Open Food Facts flagged data quality issues on this product "
            "— numbers may be off."
        )

    return MealEstimate(
        items=[item],
        total=nutrients,
        total_with_hidden=nutrients,
        overall_confidence=item.confidence,
        warnings=warnings,
    )


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


def _normalize_enums(data: dict) -> dict:
    """Normalize LLM enum outputs to valid values.

    The LLM sometimes returns descriptive strings like "good natural light"
    instead of the strict enum value "good". Map them to the nearest valid
    ImageQuality value before Pydantic validation.
    """
    _QUALITY_MAP = {"good": "good", "moderate": "moderate", "poor": "poor"}

    def _coerce_quality(val: str) -> str:
        if not isinstance(val, str):
            return "moderate"
        v = val.lower()
        for key in ("poor", "moderate", "good"):
            if key in v:
                return key
        return "moderate"

    scene = data.get("scene", {})
    if isinstance(scene, dict):
        for field in ("lighting_quality", "image_quality"):
            if field in scene:
                scene[field] = _coerce_quality(scene[field])

    return data
