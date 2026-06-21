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
from src.utils.config import Config, ENV_MODE  # <-- 1. Imported central source of truth

logger = get_logger(__name__)

def main():  # <-- 2. Removed the filepath parameter
    logger.info(f"=== Initializing Redrob Pipeline [{ENV_MODE.upper()} MODE] ===")
    logger.info(f"Reading candidates from: {Config.RAW_DATA_PATH}")
    start_time = time.time()
    
    candidate_count = 0
    for candidate in stream_candidates(Config.RAW_DATA_PATH):
        candidate_count += 1
        
    elapsed = time.time() - start_time
    logger.info(f"=== Pipeline Smoke Test Complete. Loaded {candidate_count} candidates in {elapsed:.2f}s ===")

if __name__ == "__main__":
    main()  # <-- 3. Clean, zero-argument ignition

