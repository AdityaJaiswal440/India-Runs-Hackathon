# scripts/precompute_embeddings.py
import argparse
import numpy as np
import os
import time
from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import stream_candidates
from src.features.embedder import Embedder

logger = get_logger(__name__)

def run_precompute(is_smoke_test: bool = False):
    logger.info("=== Starting Offline Precomputation ===")
    start_time = time.time()
    
    embedder = Embedder()
    candidate_ids = []
    candidate_texts = []
    
    # 1. SET THE SMOKE LIMIT
    limit = 10 if is_smoke_test else float('inf')
    if is_smoke_test:
        logger.warning("SMOKE TEST MODE ACTIVE: Capping stream at 10 records.")

    for i, candidate in enumerate(stream_candidates(Config.RAW_DATA_PATH)):
        if i >= limit:
            break
            
        candidate_ids.append(candidate["candidate_id"])
        
        headline = candidate.get("profile", {}).get("headline", "")
        summary = candidate.get("profile", {}).get("summary", "")
        career_history = candidate.get("career_history", [])
        descriptions = [job.get("description", "") for job in career_history]
        
        full_text = f"{headline} {summary} {' '.join(descriptions)}"
        candidate_texts.append(full_text)
        
    logger.info(f"Prepared {len(candidate_texts)} candidates. Vectorizing...")
    embeddings = embedder.embed_texts(candidate_texts)
    
    # 2. PROTECT THE MASTER FILE DURING SMOKE TESTS
    out_emb_path = "data/interim/smoke_embeddings.npy" if is_smoke_test else Config.EMBEDDINGS_PATH
    out_id_path = "data/interim/smoke_ids.npy" if is_smoke_test else Config.CANDIDATE_IDS_PATH

    os.makedirs(os.path.dirname(out_emb_path), exist_ok=True)
    np.save(out_emb_path, embeddings)
    np.save(out_id_path, np.array(candidate_ids))
    
    logger.info(f"Saved matrix of shape {embeddings.shape} to {out_emb_path}")
    logger.info(f"=== Complete in {time.time() - start_time:.2f}s ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a fast 10-row dry run")
    args = parser.parse_args()
    
    run_precompute(is_smoke_test=args.smoke)