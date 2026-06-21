# src/ranking/reasoning_gen.py
"""
Module: Reasoning Generator
Responsibility: Generate 1-2 sentence, fact-grounded justifications for 
candidate rankings based on Parquet feature values. No hallucination.
"""
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_reasoning(row: pd.Series) -> str:
    """Creates a templated, fact-specific reasoning string for a candidate."""
    
    # Extract facts
    yoe = row.get("yoe", 0)
    dense_sim = row.get("dense_sim_score", 0)
    bm25 = row.get("bm25_score", 0)
    behavior = row.get("behavior_score", 0)
    response_rate = row.get("response_rate", 0)
    notice = row.get("notice_period", 0)
    
    # Build semantic fit phrase
    if dense_sim > 0.6 or bm25 > 10:
        fit_phrase = "Strong semantic alignment with production ML and retrieval systems."
    else:
        fit_phrase = "Moderate alignment with JD requirements."
        
    # Build behavior phrase
    if behavior >= 0.9:
        behav_phrase = f"Highly available ({response_rate*100:.0f}% response rate, {notice}d notice)."
    elif behavior <= 0.5:
        behav_phrase = f"Availability concerns (low response rate, {notice}d notice)."
    else:
        behav_phrase = f"Behavioral signals acceptable ({response_rate*100:.0f}% response rate)."
        
    # Combine
    reasoning = f"{yoe}y experience. {fit_phrase} {behav_phrase}"
    return reasoning