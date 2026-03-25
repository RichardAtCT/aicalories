"""USDA FoodData Central API client.

Searches the USDA FDC database for food items and retrieves their
nutritional data. Prefers a local SQLite database (built by
scripts/build_db.py) for offline use. Falls back to the live API,
then to a bundled common foods file.

API docs: https://fdc.nal.usda.gov/api-guide
Rate limit: 1,000 requests per hour per API key (free tier).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path

import httpx

from .models import USDACandidate

logger = logging.getLogger(__name__)

_LOCAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "usda.db"

FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# Preferred data types in order:
# - "Survey (FNDDS)" — best for mixed dishes and common preparations
# - "SR Legacy" — USDA Standard Reference, comprehensive single ingredients
# - "Foundation" — detailed analytical data
PREFERRED_DATA_TYPES = ["Survey (FNDDS)", "SR Legacy", "Foundation"]

# Nutrient IDs we care about (USDA nutrient numbers)
NUTRIENT_MAP = {
    1008: "calories_per_100g",       # Energy (kcal)
    1003: "protein_per_100g",        # Protein
    1004: "fat_per_100g",            # Total lipid (fat)
    1005: "carbs_per_100g",          # Carbohydrate
    1079: "fiber_per_100g",          # Fiber, total dietary
    2000: "sugar_per_100g",          # Sugars, total
    1093: "sodium_per_100g",         # Sodium
    1258: "saturated_fat_per_100g",  # Fatty acids, total saturated
}


class LocalUSDAClient:
    """Offline USDA client backed by a local SQLite + FTS5 database."""

    # Data type preference: lower = better
    _TYPE_RANK = {"Survey (FNDDS)": 0, "Foundation": 1, "SR Legacy": 2}

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @staticmethod
    def _sanitize_fts(query: str) -> str:
        """Escape a query string for safe use in FTS5 MATCH expressions.

        Wraps each token in double quotes so FTS5 treats them as literal phrases
        rather than operators or column references.
        """
        import re
        # Strip FTS5 special chars, keep alphanumeric + spaces
        cleaned = re.sub(r'[^\w\s]', ' ', query, flags=re.UNICODE)
        words = [w for w in cleaned.split() if w]
        if not words:
            return '""'
        # Wrap each word in quotes → FTS5 literal phrase match per word
        return " OR ".join(f'"{w}"' for w in words)

    def _query_sync(self, query: str, max_results: int) -> list[USDACandidate]:
        """Run the FTS5 search synchronously (called via asyncio.to_thread)."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            fts_query = self._sanitize_fts(query)

            rows = conn.execute(
                """
                SELECT f.*
                FROM foods_fts fts
                JOIN foods f ON f.fdc_id = fts.rowid
                WHERE foods_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, max_results * 3),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        # Sort by data type preference, keep BM25 rank as tiebreaker
        def sort_key(r: sqlite3.Row) -> tuple[int, int]:
            idx = rows.index(r)  # original rank position
            type_rank = self._TYPE_RANK.get(r["data_type"], 9)
            return (type_rank, idx)

        rows_sorted = sorted(rows, key=sort_key)

        candidates = []
        for r in rows_sorted[:max_results]:
            if r["calories_per_100g"] <= 0:
                continue
            candidates.append(
                USDACandidate(
                    fdc_id=r["fdc_id"],
                    description=r["description"],
                    food_category=r["food_category"] or "",
                    serving_size_g=r["serving_size_g"],
                    serving_description=r["serving_description"] or "",
                    calories_per_100g=r["calories_per_100g"] or 0,
                    protein_per_100g=r["protein_per_100g"] or 0,
                    fat_per_100g=r["fat_per_100g"] or 0,
                    carbs_per_100g=r["carbs_per_100g"] or 0,
                    fiber_per_100g=r["fiber_per_100g"] or 0,
                    sugar_per_100g=r["sugar_per_100g"] or 0,
                    sodium_per_100g=r["sodium_per_100g"] or 0,
                    saturated_fat_per_100g=r["saturated_fat_per_100g"] or 0,
                )
            )
        return candidates

    async def search(self, query: str, max_results: int = 5) -> list[USDACandidate]:
        """Search the local database asynchronously."""
        return await asyncio.to_thread(self._query_sync, query, max_results)


class USDAClient:
    """Async client for the USDA FoodData Central API."""

    def __init__(
        self,
        api_key: str | None = None,
        use_fallback: bool = True,
        timeout: float = 10.0,
    ):
        self.api_key = api_key or os.environ.get("USDA_API_KEY", "")
        self.use_fallback = use_fallback
        self.timeout = timeout
        self._fallback_data: dict | None = None

        # Prefer local DB when available
        if _LOCAL_DB_PATH.exists():
            self._local: LocalUSDAClient | None = LocalUSDAClient(_LOCAL_DB_PATH)
            logger.info("Using local USDA database at %s", _LOCAL_DB_PATH)
        else:
            self._local = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[USDACandidate]:
        """Search for food items and return candidates with nutrition data.

        Args:
            query: Food description to search for, e.g.
                   "grilled chicken breast cooked"
            max_results: Maximum number of candidates to return.

        Returns:
            List of USDACandidate objects with nutrition data.
        """
        # Try local DB first
        if self._local:
            try:
                results = await self._local.search(query, max_results)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Local USDA DB search failed: {e}")

        if self.api_key:
            try:
                return await self._search_api(query, max_results)
            except Exception as e:
                logger.warning(f"USDA API search failed: {e}")
                if self.use_fallback:
                    return self._search_fallback(query, max_results)
                return []
        elif self.use_fallback:
            return self._search_fallback(query, max_results)
        else:
            logger.error("No USDA API key and fallback disabled")
            return []

    async def _search_api(
        self,
        query: str,
        max_results: int,
    ) -> list[USDACandidate]:
        """Search the live USDA FDC API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Use the search endpoint
            response = await client.post(
                f"{FDC_BASE_URL}/foods/search",
                params={"api_key": self.api_key},
                json={
                    "query": query,
                    "pageSize": max_results * 2,  # over-fetch then filter
                    "dataType": PREFERRED_DATA_TYPES,
                    "sortBy": "dataType.keyword",
                    "sortOrder": "asc",
                },
            )
            response.raise_for_status()
            data = response.json()

        candidates = []
        for food in data.get("foods", []):
            candidate = self._parse_food_item(food)
            if candidate and candidate.calories_per_100g > 0:
                candidates.append(candidate)
            if len(candidates) >= max_results:
                break

        return candidates

    def _parse_food_item(self, food: dict) -> USDACandidate | None:
        """Parse a food item from the API response into a USDACandidate."""
        try:
            nutrients = {}
            for nutrient in food.get("foodNutrients", []):
                nid = nutrient.get("nutrientId") or nutrient.get("nutrientNumber")
                # The search endpoint uses nutrientId, detail uses nutrientNumber
                if isinstance(nid, str):
                    try:
                        nid = int(nid)
                    except ValueError:
                        continue
                if nid in NUTRIENT_MAP:
                    nutrients[NUTRIENT_MAP[nid]] = nutrient.get("value", 0) or 0

            # Try to get serving info
            serving_size_g = None
            serving_desc = ""
            portions = food.get("foodPortions", []) or food.get("foodMeasures", [])
            if portions:
                first = portions[0]
                serving_size_g = first.get("gramWeight") or first.get("disseminationText")
                serving_desc = (
                    first.get("portionDescription")
                    or first.get("disseminationText")
                    or first.get("modifier", "")
                )
                if isinstance(serving_size_g, str):
                    # Sometimes it's text like "1 cup"
                    serving_desc = serving_size_g
                    serving_size_g = None

            return USDACandidate(
                fdc_id=food["fdcId"],
                description=food.get("description", ""),
                food_category=food.get("foodCategory", ""),
                serving_size_g=serving_size_g,
                serving_description=serving_desc,
                **nutrients,
            )
        except Exception as e:
            logger.debug(f"Failed to parse food item: {e}")
            return None

    def _search_fallback(
        self,
        query: str,
        max_results: int,
    ) -> list[USDACandidate]:
        """Search the bundled common foods fallback data."""
        if self._fallback_data is None:
            self._load_fallback()

        if not self._fallback_data:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for food in self._fallback_data.get("foods", []):
            desc_lower = food["description"].lower()
            # Simple word-overlap scoring
            desc_words = set(desc_lower.split())
            overlap = len(query_words & desc_words)
            # Boost for substring match
            if query_lower in desc_lower:
                overlap += 3
            elif any(w in desc_lower for w in query_words if len(w) > 3):
                overlap += 1
            if overlap > 0:
                scored.append((overlap, food))

        scored.sort(key=lambda x: x[0], reverse=True)

        candidates = []
        for _, food in scored[:max_results]:
            candidates.append(USDACandidate(**food))

        return candidates

    def _load_fallback(self) -> None:
        """Load the bundled common foods JSON file."""
        fallback_path = Path(__file__).parent.parent / "data" / "common_foods.json"
        if fallback_path.exists():
            try:
                with open(fallback_path) as f:
                    self._fallback_data = json.load(f)
                logger.info(
                    f"Loaded {len(self._fallback_data.get('foods', []))} "
                    f"fallback food entries"
                )
            except Exception as e:
                logger.error(f"Failed to load fallback data: {e}")
                self._fallback_data = {"foods": []}
        else:
            logger.warning(f"Fallback data not found at {fallback_path}")
            self._fallback_data = {"foods": []}
