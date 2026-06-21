# src/pipeline.py
"""
Main orchestrator for the Redrob Ranking Pipeline.
Approach 1: Hybrid Semantic + Heuristic Scoring
- Phase 1: Offline Embeddings (Handled later)
- Phase 2: Online Heuristics (Load data -> Extract features -> Rank -> Output CSV)
"""
import time
from src.data.loader import stream_candidates
from src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

def main(candidate_filepath: str):
    logger.info("=== Initializing Redrob Ranking Pipeline ===")
    start_time = time.time()
    
    # --- Phase 1: Data Loading (Smoke Test) ---
    # The stream_candidates function will automatically handle .gz or .jsonl
    candidate_count = 0
    for candidate in stream_candidates(candidate_filepath):
        candidate_count += 1
        
    elapsed = time.time() - start_time
    logger.info(f"=== Pipeline Smoke Test Complete. Loaded {candidate_count} candidates in {elapsed:.2f}s ===")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Fallback to a sample file path
        path = "data/raw/sample_candidates.jsonl" 
        
    main(path)