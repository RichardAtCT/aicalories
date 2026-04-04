"""Semantic search over the local USDA SQLite database.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed food descriptions
and find semantically similar USDA entries via cosine similarity.

Embeddings are pre-computed and stored as a .npz file alongside the
SQLite database.  If the file is missing, embeddings are built on first
use (takes ~30-60 s on CPU for ~13.5 k entries).

This module is imported lazily — if sentence-transformers is not installed
the rest of the codebase continues to work with FTS-only retrieval.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Paths relative to the project data/ directory
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DATA_DIR / "usda.db"
_EMBEDDINGS_PATH = _DATA_DIR / "usda_embeddings.npz"

# Model config
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _load_model():
    """Load the sentence-transformer model (lazy, cached)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build_embeddings(
    db_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[int]]:
    """Build and save embeddings for all foods in the USDA database.

    Returns:
        (embeddings, fdc_ids) — L2-normalised embedding matrix and
        corresponding FDC IDs.
    """
    db_path = db_path or _DB_PATH
    output_path = output_path or _EMBEDDINGS_PATH

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT fdc_id, description FROM foods ORDER BY fdc_id"
    ).fetchall()
    conn.close()

    fdc_ids = [r[0] for r in rows]
    descriptions = [r[1] for r in rows]

    logger.info("Encoding %d food descriptions with %s …", len(descriptions), MODEL_NAME)
    model = _load_model()
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise for cosine via dot product
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.savez_compressed(
        str(output_path),
        embeddings=embeddings,
        fdc_ids=np.array(fdc_ids, dtype=np.int64),
    )
    logger.info("Saved embeddings to %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    return embeddings, fdc_ids


class SemanticSearcher:
    """Cosine-similarity search over pre-computed USDA food embeddings."""

    def __init__(
        self,
        db_path: Path | None = None,
        embeddings_path: Path | None = None,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        self._embeddings_path = embeddings_path or _EMBEDDINGS_PATH
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._fdc_ids: np.ndarray | None = None
        self._loaded = False

    # ── lazy loading ─────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load model + embeddings on first use.  Build if missing."""
        if self._loaded:
            return

        if self._embeddings_path.exists():
            data = np.load(str(self._embeddings_path))
            self._embeddings = data["embeddings"]
            self._fdc_ids = data["fdc_ids"]
            logger.info(
                "Loaded %d pre-computed embeddings from %s",
                len(self._fdc_ids),
                self._embeddings_path,
            )
        else:
            logger.info("No pre-computed embeddings found — building now …")
            self._embeddings, fdc_ids_list = build_embeddings(
                self._db_path, self._embeddings_path
            )
            self._fdc_ids = np.array(fdc_ids_list, dtype=np.int64)

        self._model = _load_model()
        self._loaded = True

    # ── public API ───────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top-k semantically similar USDA entries.

        Each result dict contains:
            fdc_id (int), score (float 0–1)
        """
        self._ensure_loaded()

        # Encode query (L2-normalised)
        query_emb = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Cosine similarity via dot product (both sides L2-normalised)
        scores = (self._embeddings @ query_emb.T).squeeze()

        # Top-k indices
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            results.append({
                "fdc_id": int(self._fdc_ids[idx]),
                "score": round(float(scores[idx]), 4),
            })
        return results
