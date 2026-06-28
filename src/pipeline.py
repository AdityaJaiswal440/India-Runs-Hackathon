"""Ranking pipeline entry point.

Orchestrates the full offline ranking run:
  1. Load pre-computed embeddings from artifacts/
  2. Stream candidates from data/raw/candidates.jsonl
  3. Score and rank using the hybrid ranker
  4. Write submission.csv with ASCII-safe reasoning strings
"""

import csv
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the import path when invoked directly
_src_dir = os.path.abspath(os.path.dirname(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

_root_dir = os.path.abspath(os.path.join(_src_dir, ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# Canonical submission uses bi-encoder only (deterministic, network-off)
os.environ.setdefault("RANKER_USE_CROSS_ENCODER", "0")
os.environ.setdefault("RANKER_REQUIRE_EMBEDDINGS", "1")

from challenge.redrob_ranker import rank_candidates  # noqa: E402
from src.utils.logger import get_logger              # noqa: E402

logger = get_logger("pipeline")

_NON_ASCII_MAP = str.maketrans({
    "\u2014": "-",   # em dash
    "\u00b7": "-",   # middle dot
    "\u2026": "...", # ellipsis
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
})


def _ascii_reasoning(text: str) -> str:
    """Replace common Unicode typographic characters then hard-enforce ASCII."""
    return text.translate(_NON_ASCII_MAP).encode("ascii", "ignore").decode("ascii")


def main() -> None:
    start_time = time.time()
    logger.info("=== Redrob Ranking Pipeline — starting ===")

    candidates_path = "data/raw/candidates.jsonl"
    out_path = Path("submission.csv")

    logger.info(f"Ranking candidates from: {candidates_path}")
    top_candidates = rank_candidates(candidates_path, top_k=100)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, row in enumerate(top_candidates):
            writer.writerow([
                row.candidate_id,
                i + 1,
                f"{row.score:.6f}",
                _ascii_reasoning(row.reasoning),
            ])

    elapsed = time.time() - start_time
    logger.info(
        f"=== Pipeline complete in {elapsed:.2f}s — top candidate: {top_candidates[0].candidate_id} "
        f"score {top_candidates[0].score:.6f} ==="
    )


if __name__ == "__main__":
    main()