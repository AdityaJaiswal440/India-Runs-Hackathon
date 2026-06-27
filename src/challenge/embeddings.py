"""Bi-encoder career embeddings — precomputed artifact with graceful fallback."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from challenge.jd_config import JD_DOCUMENT
from challenge.text_match import norm_text

_BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings"


def candidate_text(profile: Dict, history: list) -> str:
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        *[h.get("description", "") for h in history[:4]],
    ]
    return norm_text(" ".join(p for p in parts if p))[:4000]


class EmbeddingStore:
    """Load precomputed JD + candidate embeddings; cosine lookup by candidate_id."""

    def __init__(self, root: Path | None = None):
        self.root = root or _DEFAULT_DIR
        self._ids: Dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._jd: np.ndarray | None = None
        self.available = False
        self._load()

    def _load(self) -> None:
        ids_path = self.root / "candidate_ids.json"
        if not ids_path.exists():
            return

        matrix: np.ndarray | None = None
        jd: np.ndarray | None = None

        npz_path = self.root / "embeddings.fp16.npz"
        if npz_path.exists():
            try:
                with np.load(npz_path) as z:
                    matrix = z["embeddings"].astype(np.float32)
                    jd = z["jd_embedding"].astype(np.float32)
            except (OSError, ValueError, KeyError):
                matrix = jd = None

        if matrix is None:
            emb_path = self.root / "embeddings.npy"
            jd_path = self.root / "jd_embedding.npy"
            if not (emb_path.exists() and jd_path.exists()):
                return
            try:
                matrix = np.load(emb_path).astype(np.float32)
                jd = np.load(jd_path).astype(np.float32)
            except OSError:
                return

        try:
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if matrix is None or jd is None or len(ids) != matrix.shape[0]:
                return
            self._matrix = matrix
            self._jd = jd
            self._ids = {cid: i for i, cid in enumerate(ids)}
            self.available = True
        except (ValueError, json.JSONDecodeError):
            self.available = False

    def cosine_vs_jd(self, candidate_id: str) -> Optional[float]:
        if not self.available or self._matrix is None or self._jd is None:
            return None
        idx = self._ids.get(candidate_id)
        if idx is None:
            return None
        vec = self._matrix[idx]
        jd = self._jd
        dot = float(np.dot(vec, jd))
        na = float(np.linalg.norm(vec))
        nb = float(np.linalg.norm(jd))
        if na == 0 or nb == 0:
            return 0.0
        # Map cosine [-1,1] → [0,1]
        cos = dot / (na * nb)
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))

    @staticmethod
    def model_name() -> str:
        return _BI_ENCODER_MODEL

    def describe(self) -> str:
        if self.available:
            npz = self.root / "embeddings.fp16.npz"
            artifact = "embeddings.fp16.npz" if npz.exists() else "embeddings.npy"
            return f"LOADED ({_BI_ENCODER_MODEL}, {artifact})"
        return "MISSING — TF-IDF JD proxy fallback (non-canonical ranking path)"

    @staticmethod
    def jd_source_text() -> str:
        return JD_DOCUMENT


def embeddings_required() -> bool:
    """Default ON: refuse silent TF-IDF fallback during submission reproduction."""
    return os.environ.get("RANKER_REQUIRE_EMBEDDINGS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def guard_canonical_embeddings(store: EmbeddingStore) -> None:
    """Fail or warn when committed embeddings are not loaded."""
    if store.available:
        return
    msg = (
        "Committed embeddings not loaded (expected data/embeddings/embeddings.fp16.npz). "
        "Ranking would use TF-IDF fallback and produce a DIFFERENT submission.csv. "
        "Mount data/embeddings/ or run ./scripts/reproduce_ranking.sh. "
        "Set RANKER_REQUIRE_EMBEDDINGS=0 only for ablation/diagnostics."
    )
    if embeddings_required():
        raise RuntimeError(msg)
    print(f"WARNING: {msg}", file=sys.stderr)