#!/usr/bin/env python3
"""Build a local USDA FoodData Central SQLite database for offline calorie lookups.

Downloads FNDDS (Survey), Foundation Foods, and SR Legacy JSON datasets from
the USDA and stores them in data/usda.db with FTS5 full-text search.

Usage:
    python scripts/build_db.py

The database is ~50MB and replaces live API calls automatically once present.
"""

import io
import json
import os
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "usda.db"

DATASETS = [
    {
        "name": "FNDDS (Survey Foods)",
        "url": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_json_2024-10-31.zip",
        "key": "SurveyFoods",
    },
    {
        "name": "Foundation Foods",
        "url": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_json_2024-10-31.zip",
        "key": "FoundationFoods",
    },
    {
        "name": "SR Legacy",
        "url": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_json_2021-10-28.zip",
        "key": "SRLegacyFoods",
    },
]

# USDA nutrient IDs we care about
NUTRIENT_IDS = {
    1008: "calories_per_100g",       # Energy (kcal)
    1003: "protein_per_100g",        # Protein
    1004: "fat_per_100g",            # Total lipid (fat)
    1005: "carbs_per_100g",          # Carbohydrate, by difference
    1079: "fiber_per_100g",          # Fiber, total dietary
    2000: "sugar_per_100g",          # Sugars, total (also try 1063)
    1063: "sugar_per_100g",          # Sugars, total (alternate ID)
    1093: "sodium_per_100g",         # Sodium
    1258: "saturated_fat_per_100g",  # Fatty acids, total saturated
}


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS foods_fts;
        DROP TABLE IF EXISTS foods;

        CREATE TABLE foods (
            fdc_id              INTEGER PRIMARY KEY,
            description         TEXT NOT NULL,
            food_category       TEXT DEFAULT "",
            data_type           TEXT DEFAULT "",
            serving_size_g      REAL,
            serving_description TEXT DEFAULT "",
            calories_per_100g   REAL DEFAULT 0,
            protein_per_100g    REAL DEFAULT 0,
            fat_per_100g        REAL DEFAULT 0,
            carbs_per_100g      REAL DEFAULT 0,
            fiber_per_100g      REAL DEFAULT 0,
            sugar_per_100g      REAL DEFAULT 0,
            sodium_per_100g     REAL DEFAULT 0,
            saturated_fat_per_100g REAL DEFAULT 0
        );

        CREATE VIRTUAL TABLE foods_fts USING fts5(
            description,
            food_category,
            content=foods,
            content_rowid=fdc_id,
            tokenize="unicode61"
        );
    """)


def extract_nutrients(food_nutrients: list) -> dict:
    result = {}
    for fn in food_nutrients:
        nutrient = fn.get("nutrient", {}) or fn.get("foodNutrient", {})
        nid = nutrient.get("id") or fn.get("nutrientId")
        amount = fn.get("amount", 0) or fn.get("value", 0) or 0
        if nid in NUTRIENT_IDS:
            col = NUTRIENT_IDS[nid]
            # Don't overwrite if already set (prefer first occurrence)
            if col not in result:
                result[col] = float(amount)
    return result


def parse_food(food: dict) -> dict | None:
    fdc_id = food.get("fdcId")
    if not fdc_id:
        return None

    description = food.get("description", "").strip()
    if not description:
        return None

    # Food category — varies by dataset
    cat = food.get("foodCategory") or food.get("wweiaFoodCategory") or {}
    if isinstance(cat, dict):
        food_category = cat.get("description", "") or cat.get("wweiaFoodCategoryDescription", "")
    else:
        food_category = str(cat)

    # Serving size
    portions = food.get("foodPortions", [])
    serving_size_g = None
    serving_description = ""
    if portions:
        p = portions[0]
        gram_weight = p.get("gramWeight")
        if gram_weight:
            serving_size_g = float(gram_weight)
            serving_description = p.get("portionDescription", "") or p.get("modifier", "")

    nutrients = extract_nutrients(food.get("foodNutrients", []))

    return {
        "fdc_id": fdc_id,
        "description": description,
        "food_category": food_category or "",
        "serving_size_g": serving_size_g,
        "serving_description": serving_description,
        **{k: nutrients.get(k, 0) for k in [
            "calories_per_100g", "protein_per_100g", "fat_per_100g",
            "carbs_per_100g", "fiber_per_100g", "sugar_per_100g",
            "sodium_per_100g", "saturated_fat_per_100g",
        ]},
    }


def download_and_parse(dataset: dict) -> list[dict]:
    name = dataset["name"]
    url = dataset["url"]
    key = dataset["key"]

    print(f"\n📥  Downloading {name}...")
    print(f"    {url}")

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunks = []
        while True:
            chunk = resp.read(1024 * 256)  # 256KB chunks
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct:.0f}%)", end="", flush=True)
        data = b"".join(chunks)
    print(f"\r    ✅  Downloaded {len(data) // 1024 // 1024}MB            ")

    print(f"    Parsing JSON...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_files:
            print(f"    ⚠️  No JSON file found in zip")
            return []
        with zf.open(json_files[0]) as jf:
            raw = json.load(jf)

    foods_raw = raw.get(key, [])
    print(f"    Parsing {len(foods_raw):,} foods...")

    parsed = []
    for food in foods_raw:
        row = parse_food(food)
        if row:
            row["data_type"] = key
            parsed.append(row)

    print(f"    ✅  Parsed {len(parsed):,} foods")
    return parsed


def insert_batch(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.executemany("""
        INSERT OR REPLACE INTO foods (
            fdc_id, description, food_category, data_type,
            serving_size_g, serving_description,
            calories_per_100g, protein_per_100g, fat_per_100g,
            carbs_per_100g, fiber_per_100g, sugar_per_100g,
            sodium_per_100g, saturated_fat_per_100g
        ) VALUES (
            :fdc_id, :description, :food_category, :data_type,
            :serving_size_g, :serving_description,
            :calories_per_100g, :protein_per_100g, :fat_per_100g,
            :carbs_per_100g, :fiber_per_100g, :sugar_per_100g,
            :sodium_per_100g, :saturated_fat_per_100g
        )
    """, rows)


def rebuild_fts(conn: sqlite3.Connection) -> None:
    print("\n🔍  Building FTS5 index...")
    conn.execute("INSERT INTO foods_fts(foods_fts) VALUES('rebuild')")
    print("    ✅  FTS5 index ready")


def sanity_check(conn: sqlite3.Connection) -> None:
    print("\n🧪  Sanity check queries:")
    queries = ["grilled chicken breast", "french fries", "brown rice cooked", "whole milk"]
    for q in queries:
        rows = conn.execute("""
            SELECT f.description, f.calories_per_100g, f.data_type
            FROM foods_fts
            JOIN foods f ON foods_fts.rowid = f.fdc_id
            WHERE foods_fts MATCH ?
            ORDER BY rank
            LIMIT 1
        """, (q,)).fetchall()
        if rows:
            desc, cals, dtype = rows[0]
            print(f"    '{q}' → {desc[:60]} ({cals:.0f} kcal/100g) [{dtype}]")
        else:
            print(f"    '{q}' → no results")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        print(f"⚠️  Existing database found at {DB_PATH}")
        ans = input("   Rebuild from scratch? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
        DB_PATH.unlink()

    print(f"📦  Building USDA database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    create_schema(conn)
    conn.commit()

    total_foods = 0
    for dataset in DATASETS:
        try:
            rows = download_and_parse(dataset)
            if rows:
                insert_batch(conn, rows)
                conn.commit()
                total_foods += len(rows)
        except Exception as e:
            print(f"\n    ⚠️  Failed to process {dataset['name']}: {e}")
            continue

    rebuild_fts(conn)
    conn.commit()

    sanity_check(conn)

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n✅  Database ready: {total_foods:,} foods, {size_mb:.1f}MB → {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
