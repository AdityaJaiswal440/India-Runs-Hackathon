# src/pipeline.py
"""
Module: Pipeline Orchestrator (Online Phase)
Responsibility: Load precomputed feature table, filter honeypots/disqualifiers,
score candidates, sort, generate reasoning, and write final submission CSV.
"""
import os
from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

import time
import pandas as pd
from src.utils.config import Config
from src.utils.logger import get_logger
from src.ranking.heuristic_scorer import compute_scores
from src.ranking.reasoning_gen import generate_reasoning

logger = get_logger("pipeline")

def main():
    logger.info("=== Initializing Redrob Ranking Pipeline (Online Phase) ===")
    start_time = time.time()
    
    # 1. Load Feature Table
    logger.info(f"Loading features from {Config.FEATURES_TABLE_PATH}...")
    if not os.path.exists(Config.FEATURES_TABLE_PATH):
        logger.error("Feature table not found! Run `make precompute` first.")
        return
        
    df = pd.read_parquet(Config.FEATURES_TABLE_PATH)
    logger.info(f"Loaded {len(df)} candidates.")
    
    # 2. Filter First (Hard Disqualifiers)
    initial_count = len(df)
    df = df[~df["is_honeypot"]].copy()
    df = df[~df["is_consulting_only"]].copy()
    filtered_count = len(df)
    logger.info(f"Filtered out {initial_count - filtered_count} honeypots/consulting-only candidates. {filtered_count} remaining.")
    
    # 3. Compute Scores
    df = compute_scores(df)
    
    # 4. Sort and Take Top 100
    df = df.sort_values(by="final_score", ascending=False).head(100).reset_index(drop=True)
    
    # 5. Assign Ranks and Generate Reasoning
    df["rank"] = df.index + 1
    df["reasoning"] = df.apply(generate_reasoning, axis=1)
    
    # 6. Format Output CSV
    output_df = df[["candidate_id", "rank", "final_score", "reasoning"]]
    output_df.rename(columns={"final_score": "score"}, inplace=True)
    
    output_path = "submission.csv"
    output_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"=== Pipeline Complete in {elapsed:.2f}s ===")
    logger.info(f"Submission saved to {output_path}. Top rank: {output_df.iloc[0]['candidate_id']}")

if __name__ == "__main__":
    main()