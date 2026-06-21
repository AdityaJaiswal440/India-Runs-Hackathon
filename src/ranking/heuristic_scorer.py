# src/ranking/heuristic_scorer.py
"""
Module: Heuristic Scorer
Responsibility: Apply the weighted composite scoring formula to the 
feature table. Normalizes BM25, calculates penalties, and returns final scores.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the final composite score for each candidate.
    Formula: (Dense_Sim + BM25_Norm) * Behavior_Score - YoE_Penalty
    """
    logger.info("Computing composite scores...")
    
    # 1. Normalize BM25 scores to 0.0 - 1.0 range
    # Reshape is needed because MinMaxScaler expects 2D array
    bm25_scaler = MinMaxScaler()
    df["bm25_norm"] = bm25_scaler.fit_transform(df["bm25_score"].values.reshape(-1, 1))
    
    # 2. Calculate Raw Fit (Equal weight to Dense and BM25 for now)
    df["raw_fit"] = (0.6 * df["dense_sim_score"]) + (0.4 * df["bm25_norm"])
    
    # 3. Apply Behavioral Multiplier
    # This instantly down-weights perfect-on-paper but disengaged candidates
    df["adjusted_fit"] = df["raw_fit"] * df["behavior_score"]
    
    # 4. Apply YoE Penalty
    # Subtracts a flat 0.05 for every year outside the 5-9 sweet spot
    df["final_score"] = df["adjusted_fit"] - (df["yoe_distance"] * 0.05)
    
    # Clamp scores to a minimum of 0.0
    df["final_score"] = df["final_score"].clip(lower=0.0)
    
    logger.info("Score computation complete.")
    return df