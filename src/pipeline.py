# src/pipeline.py
"""
Pipeline Orchestrator for 100% RecruitGPT X parity.
"""
import os
import sys
import csv
from pathlib import Path

# Add src/ and project root to path
src_dir = os.path.abspath(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

root_dir = os.path.abspath(os.path.join(src_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Competitor uses these env variables
os.environ["RANKER_USE_CROSS_ENCODER"] = "0"
os.environ["RANKER_REQUIRE_EMBEDDINGS"] = "1"

from challenge.redrob_ranker import rank_candidates
from challenge.embeddings import EmbeddingStore
from src.utils.logger import get_logger

logger = get_logger("pipeline")

def main():
    logger.info("=== Initializing Redrob Ranking Pipeline (Online Phase — Parity V5) ===")
    
    candidates_path = "data/raw/candidates.jsonl"
    out_path = "submission.csv"
    
    logger.info(f"Ranking candidates from: {candidates_path}")
    top_candidates = rank_candidates(candidates_path, top_k=100)
    
    logger.info(f"Writing top {len(top_candidates)} rows to {out_path} with 6-decimal score precision...")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, row in enumerate(top_candidates):
            reasoning = row.reasoning
            # Surgically replace common non-ASCII characters to preserve text flow
            reasoning = reasoning.replace("—", "-")
            reasoning = reasoning.replace("·", "-")
            reasoning = reasoning.replace("…", "...")
            reasoning = reasoning.replace("“", '"')
            reasoning = reasoning.replace("”", '"')
            reasoning = reasoning.replace("’", "'")
            reasoning = reasoning.replace("‘", "'")
            # Strict ASCII enforcement fallback
            reasoning = reasoning.encode("ascii", "ignore").decode("ascii")
            
            w.writerow([row.candidate_id, i + 1, f"{row.score:.6f}", reasoning])
            
    logger.info(f"=== Pipeline Complete. Top candidate: {top_candidates[0].candidate_id} with score {top_candidates[0].score:.6f} ===")

if __name__ == "__main__":
    main()