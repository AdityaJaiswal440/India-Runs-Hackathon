# scripts/precompute.py
"""
Module: Offline Precomputation Pipeline
Responsibility: Read raw candidates, extract heuristics, check honeypots, 
compute BM25 & Dense Embedding scores, and consolidate everything into a 
single Parquet feature table for the online ranking phase.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import stream_candidates
from src.features.embedder import Embedder
from src.features.honeypot import is_honeypot
from src.features.heuristic_extractor import extract_features

logger = get_logger("Offline_Precompute")

def simple_tokenize(text: str) -> list:
    """Basic regex tokenizer for BM25: lowercases and strips punctuation."""
    return re.findall(r'\b\w+\b', text.lower())

def run_precompute():
    logger.info("=== Starting Offline Precomputation Phase ===")
    start_time = time.time()
    
    # 1. Load the Requirements-only JD text
    logger.info(f"Loading JD text from: {Config.JD_REQUIREMENTS_PATH}")
    with open(Config.JD_REQUIREMENTS_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()
    jd_tokens = simple_tokenize(jd_text)

    # 2. Initialize Models
    embedder = Embedder()
    
    # 3. Stream and process candidates
    candidate_ids = []
    candidate_texts = []
    features_list = []
    
    logger.info("Streaming candidates, extracting features and honeypots...")
    for candidate in stream_candidates(Config.RAW_DATA_PATH):
        candidate_ids.append(candidate["candidate_id"])
        
        # Extract text for IR
        profile = candidate.get("profile", {})
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        career_history = candidate.get("career_history", [])
        descriptions = [job.get("description", "") for job in career_history]
        full_text = f"{headline} {summary} {' '.join(descriptions)}"
        candidate_texts.append(full_text)
        
        # Extract structured features
        feats = extract_features(candidate)
        feats["candidate_id"] = candidate["candidate_id"]
        feats["is_honeypot"] = is_honeypot(candidate)
        features_list.append(feats)

    logger.info(f"Processed {len(candidate_ids)} candidates. Starting BM25 and Embeddings...")

    # 4. Compute BM25 Scores
    logger.info("Tokenizing corpus for BM25...")
    tokenized_corpus = [simple_tokenize(text) for text in candidate_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(jd_tokens)
    logger.info("BM25 scoring complete.")

    # 5. Compute Dense Embeddings
    logger.info("Generating dense embeddings (this may take a few minutes)...")
    candidate_vectors = embedder.embed_texts(candidate_texts)
    jd_vector = embedder.embed_texts([jd_text])[0] # Shape: (384,)
    
    # Calculate cosine similarity via dot product 
    # (candidate_vectors is already normalized by sentence-transformers)
    dense_scores = np.dot(candidate_vectors, jd_vector)
    logger.info("Dense embeddings and similarity scoring complete.")

    # 6. Consolidate into DataFrame
    df = pd.DataFrame(features_list)
    df["bm25_score"] = bm25_scores
    df["dense_sim_score"] = dense_scores
    
    # Reorder columns for readability
    cols = ["candidate_id", "is_honeypot", "is_consulting_only", "yoe", "yoe_distance", 
            "behavior_score", "response_rate", "notice_period", "bm25_score", "dense_sim_score"]
    df = df[cols]

    # 7. Save to Parquet
    os.makedirs(os.path.dirname(Config.FEATURES_TABLE_PATH), exist_ok=True)
    logger.info(f"Saving feature table to {Config.FEATURES_TABLE_PATH}...")
    df.to_parquet(Config.FEATURES_TABLE_PATH, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"=== Precomputation Complete in {elapsed:.2f}s ===")
    logger.info(f"Feature table shape: {df.shape}")

if __name__ == "__main__":
    run_precompute()