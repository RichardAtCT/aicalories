#!/usr/bin/env python3
"""Pre-compute semantic embeddings for the local USDA SQLite database.

Usage:
    python scripts/build_embeddings.py [--db data/usda.db] [--out data/usda_embeddings.npz]

Requires: pip install sentence-transformers numpy
Takes ~30-60 s on CPU for ~13.5 k entries.  Output is ~20 MB compressed.
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calorie_estimator.semantic import build_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build USDA food embeddings")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/usda.db"),
        help="Path to the USDA SQLite database",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/usda_embeddings.npz"),
        help="Output path for the embeddings file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size (default: 256)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    embeddings, fdc_ids = build_embeddings(args.db, args.out, args.batch_size)
    print(f"Done. {len(fdc_ids)} embeddings → {args.out}")


if __name__ == "__main__":
    main()
