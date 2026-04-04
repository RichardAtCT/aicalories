"""USDA FoodData Central API client.

Searches the USDA FDC database for food items and retrieves their
nutritional data. Prefers a local SQLite database (built by
scripts/build_db.py) for offline use. Falls back to the live API,
then to a bundled common foods file.

Supports hybrid retrieval: FTS5 (lexical) + sentence-transformer
(semantic) with merged/deduped candidate lists.

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

# Lazy import — semantic search is optional
_semantic_available: bool | None = None
_semantic_searcher = None

logger = logging.getLogger(__name__)


def _get_semantic_searcher():
    """Return a SemanticSearcher singleton, or None if unavailable."""
    global _semantic_available, _semantic_searcher
    if _semantic_available is False:
        return None
    try:
        from .semantic import SemanticSearcher

        if _semantic_searcher is None:
            _semantic_searcher = SemanticSearcher()
        _semantic_available = True
        return _semantic_searcher
    except Exception as e:
        logger.info("Semantic search unavailable: %s", e)
        _semantic_available = False
        return None

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
    """Offline USDA client backed by a local SQLite + FTS5 database.

    Supports hybrid retrieval: FTS5 lexical search + optional semantic
    search via sentence-transformers.  When both are available, results
    are merged and deduplicated — semantic hits improve recall while FTS
    handles exact-match precision.
    """

    # Data type preference: lower = better
    _TYPE_RANK = {"Survey (FNDDS)": 0, "Foundation": 1, "SR Legacy": 2}

    def __init__(self, db_path: Path, enable_semantic: bool = True) -> None:
        self._db_path = db_path
        self._enable_semantic = enable_semantic

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
            c = self._build_candidate_from_row(r)
            if c is not None:
                candidates.append(c)
        return candidates

    def _fetch_candidates_by_ids(
        self, fdc_ids: list[int],
    ) -> dict[int, sqlite3.Row]:
        """Fetch food rows by FDC ID for semantic hits."""
        if not fdc_ids:
            return {}
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" * len(fdc_ids))
            rows = conn.execute(
                f"SELECT * FROM foods WHERE fdc_id IN ({placeholders})",
                fdc_ids,
            ).fetchall()
        finally:
            conn.close()
        return {r["fdc_id"]: r for r in rows}

    def _build_candidate_from_row(self, row: sqlite3.Row) -> USDACandidate | None:
        """Build a USDACandidate from a SQLite row, filtering zero-calorie entries."""
        if (row["calories_per_100g"] or 0) <= 0:
            return None
        return USDACandidate(
            fdc_id=row["fdc_id"],
            description=row["description"],
            food_category=row["food_category"] or "",
            serving_size_g=row["serving_size_g"],
            serving_description=row["serving_description"] or "",
            calories_per_100g=row["calories_per_100g"] or 0,
            protein_per_100g=row["protein_per_100g"] or 0,
            fat_per_100g=row["fat_per_100g"] or 0,
            carbs_per_100g=row["carbs_per_100g"] or 0,
            fiber_per_100g=row["fiber_per_100g"] or 0,
            sugar_per_100g=row["sugar_per_100g"] or 0,
            sodium_per_100g=row["sodium_per_100g"] or 0,
            saturated_fat_per_100g=row["saturated_fat_per_100g"] or 0,
        )

    def _hybrid_search_sync(
        self, query: str, max_results: int,
    ) -> list[USDACandidate]:
        """Hybrid FTS + semantic search, merged via round-robin interleave.

        Strategy: collect candidates from both FTS and semantic, then
        interleave them (FTS first, then semantic) while deduplicating.
        This ensures high-confidence semantic hits appear even when FTS
        returns noisy matches (e.g. "bell pepper" → "TACO BELL").
        """
        # --- FTS results ---
        fts_candidates = self._query_sync(query, max_results)

        # --- Semantic results ---
        sem_candidates: list[USDACandidate] = []
        sem_scores: dict[int, float] = {}
        if self._enable_semantic:
            searcher = _get_semantic_searcher()
            if searcher is not None:
                try:
                    sem_results = searcher.search(query, top_k=max_results * 2)
                    all_sem_ids = [
                        r["fdc_id"] for r in sem_results if r["score"] >= 0.25
                    ]
                    if all_sem_ids:
                        rows_by_id = self._fetch_candidates_by_ids(all_sem_ids)
                        for r in sem_results:
                            fdc_id = r["fdc_id"]
                            if fdc_id in rows_by_id:
                                c = self._build_candidate_from_row(rows_by_id[fdc_id])
                                if c is not None:
                                    sem_candidates.append(c)
                                    sem_scores[fdc_id] = r["score"]
                except Exception as e:
                    logger.warning("Semantic search failed, using FTS only: %s", e)

        # --- Interleave merge ---
        # Round-robin: take from FTS and semantic alternately, dedup by fdc_id.
        # This gives both retrieval paths a fair shot rather than one dominating.
        merged: list[USDACandidate] = []
        seen: set[int] = set()
        fi, si = 0, 0
        while len(merged) < max_results and (fi < len(fts_candidates) or si < len(sem_candidates)):
            # Take one from FTS
            while fi < len(fts_candidates):
                c = fts_candidates[fi]
                fi += 1
                if c.fdc_id not in seen:
                    merged.append(c)
                    seen.add(c.fdc_id)
                    break
            # Take one from semantic
            while si < len(sem_candidates):
                c = sem_candidates[si]
                si += 1
                if c.fdc_id not in seen:
                    merged.append(c)
                    seen.add(c.fdc_id)
                    break

        logger.debug(
            "Hybrid search for %r: %d FTS + %d semantic → %d merged",
            query, len(fts_candidates), len(sem_candidates), len(merged),
        )
        return merged[:max_results]

    async def search(self, query: str, max_results: int = 5) -> list[USDACandidate]:
        """Search the local database asynchronously (hybrid FTS + semantic)."""
        return await asyncio.to_thread(self._hybrid_search_sync, query, max_results)


class USDAClient:
    """Async client for the USDA FoodData Central API."""

    def __init__(
        self,
        api_key: str | None = None,
        use_fallback: bool = True,
        timeout: float = 10.0,
        enable_semantic: bool = True,
    ):
        self.api_key = api_key or os.environ.get("USDA_API_KEY", "")
        self.use_fallback = use_fallback
        self.timeout = timeout
        self._fallback_data: dict | None = None

        # Prefer local DB when available
        if _LOCAL_DB_PATH.exists():
            self._local: LocalUSDAClient | None = LocalUSDAClient(
                _LOCAL_DB_PATH, enable_semantic=enable_semantic,
            )
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
