# Data Directory

## Local USDA Database

The calorie estimator can use a local SQLite database for offline USDA food lookups.

### Building the database

```bash
python scripts/build_db.py
```

This downloads USDA FoodData Central datasets (~300 MB download) and builds `usda.db` (~50 MB).

- **FNDDS (Survey Foods)** — common mixed dishes and preparations
- **Foundation Foods** — detailed analytical data for single ingredients
- **SR Legacy** — USDA Standard Reference, comprehensive single ingredients

### Usage

Once `usda.db` exists in this directory, the `CalorieEstimator` automatically uses it.
No USDA API key is needed for local lookups. If the DB is absent, the estimator falls
back to the live API (requires `USDA_API_KEY`) or the bundled `common_foods.json`.

### Note

`usda.db` is git-ignored because it's too large to commit. Each developer runs the
build script once to create their local copy.
