# scripts/precompute.py
"""
Module: Offline Precomputation Pipeline (V3 Architecture)
Responsibility: Stream candidates, find temporal anchor, detect honeypots,
compute lexical and dense embeddings, execute Reciprocal Rank Fusion,
and persist all intermediate artifacts safely to disk.
"""

from pathlib import Path
import os
import sys
import gc
import pickle
import datetime
import time
import json
import numpy as np
import pandas as pd
from collections import Counter
from rank_bm25 import BM25Okapi

sys.path.append(str(Path.cwd()))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.loader import stream_candidates
from src.features.embedder import Embedder
from src.challenge.honeypot import is_honeypot, safe_date
from src.features.heuristic_extractor import (
    compute_skill_match_score,
    compute_skill_gap_coverage,
    compute_career_score,
    compute_experience_fit,
    compute_location_score,
    compute_education_score,
    compute_engagement_score,
    JD_TAXONOMY,
    TIER_1_SKILLS
)

logger = get_logger("Offline_Precompute_V3")

def simple_tokenize(text: str) -> list:
    """Basic regex tokenizer for BM25: lowercases and strips punctuation."""
    return re.findall(r'\b\w+\b', text.lower()) if text else []

import re

def compute_dataset_today(filepath: str) -> str:
    """
    Computes the dataset-derived TODAY temporal anchor.
    Finds the maximum last_active_date clamped against the real-world current date.
    """
    max_date = None
    real_world_today = datetime.date.today()
    for candidate in stream_candidates(filepath):
        signals = candidate.get("redrob_signals", {}) or {}
        last_active = signals.get("last_active_date")
        if last_active:
            try:
                dt = datetime.date.fromisoformat(last_active)
                if dt <= real_world_today:
                    if max_date is None or dt > max_date:
                        max_date = dt
            except (ValueError, TypeError):
                continue
    if max_date is None:
        return real_world_today.isoformat()
    return max_date.isoformat()

def build_profile_signature(candidate: dict) -> tuple:
    """
    Constructs a unique profile signature tuple for duplicate/clone detection.
    """
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    
    yoe = profile.get("years_of_experience")
    title = (profile.get("current_title") or "").lower().strip()
    company = (profile.get("current_company") or "").lower().strip()
    loc = (profile.get("location") or "").lower().strip()
    country = (profile.get("country") or "").lower().strip()
    csize = (profile.get("current_company_size") or "").lower().strip()
    ind = (profile.get("current_industry") or "").lower().strip()
    
    skill_names = tuple(sorted(str(s.get("name")).lower().strip() for s in skills if s and s.get("name")))
    edu_details = tuple(sorted((str(e.get("institution")).lower().strip(), str(e.get("degree")).lower().strip()) for e in education if e))
    
    return (yoe, title, company, loc, country, csize, ind, skill_names, edu_details)

def build_candidate_text(candidate: dict) -> str:
    """
    Builds the text input for candidate dense embedding.
    """
    profile = candidate.get("profile", {}) or {}
    headline = profile.get("headline") or ""
    current_title = profile.get("current_title") or ""
    
    skills = candidate.get("skills", []) or []
    skill_names = [s.get("name") for s in skills if s and s.get("duration_months", 0) > 0 and s.get("name")]
    skills_part = "Skills: " + " ".join(skill_names) if skill_names else ""
    
    career_history = candidate.get("career_history", []) or []
    descriptions = [job.get("description") or "" for job in career_history if job]
    career_text = " ".join(descriptions)
    if len(career_text) > 300:
        career_text = career_text[:300]
        
    parts = [headline, current_title, skills_part, career_text]
    return " ".join([p for p in parts if p.strip()])

def token_generator(filepath: str):
    """
    Generator pipeline yielding tokenized summaries and career histories for BM25.
    """
    for candidate in stream_candidates(filepath):
        profile = candidate.get("profile", {}) or {}
        summary = profile.get("summary") or ""
        career_history = candidate.get("career_history", []) or []
        descriptions = [job.get("description") or "" for job in career_history if job]
        full_text = f"{summary} {' '.join(descriptions)}"
        yield simple_tokenize(full_text)

def compute_doc_frequencies(filepath: str) -> Counter:
    """
    Computes document frequencies per token across the entire candidate pool using the generator.
    """
    doc_freq = Counter()
    for tokens in token_generator(filepath):
        doc_freq.update(set(tokens))
    return doc_freq

def compute_career_keyword_scores(filepath: str) -> np.ndarray:
    """
    Computes the career description keyword score with a saturation cap of 2 per unique skill.
    """
    scores = []
    for candidate in stream_candidates(filepath):
        career_history = candidate.get("career_history", []) or []
        descriptions = [job.get("description") or "" for job in career_history if job]
        full_desc = " ".join(descriptions).lower()
        
        score_sum = 0
        for skill in TIER_1_SKILLS:
            count = full_desc.count(skill.lower())
            score_sum += min(count, 2)
        scores.append(float(score_sum))
    return np.array(scores, dtype=np.float32)

def run_precompute():
    logger.info("=== Starting Offline Precomputation Phase (V3) ===")
    start_time = time.time()
    
    # Ensure directories exist
    os.makedirs(Config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    
    # -----------------------------------------------------------------
    # Step 1: First Pass - Temporal Anchor & Clone Registry
    # -----------------------------------------------------------------
    logger.info("First Pass: Deriving DATASET_TODAY and identifying clones...")
    
    # Derive TODAY
    today_str = compute_dataset_today(Config.RAW_DATA_PATH)
    logger.info(f"Derived DATASET_TODAY: {today_str}")
    
    # Write temporal anchor
    with open(os.path.join(Config.ARTIFACTS_DIR, "dataset_today.txt"), "w", encoding="utf-8") as f:
        f.write(today_str)
        
    # Update Config in memory
    Config.TODAY = datetime.date.fromisoformat(today_str)
    
    # Registry duplicate profile hashes
    profile_to_ids = {}
    for candidate in stream_candidates(Config.RAW_DATA_PATH):
        cid = candidate.get("candidate_id")
        if cid:
            sig = build_profile_signature(candidate)
            profile_to_ids.setdefault(sig, []).append(cid)
            
    clone_ids = set()
    for sig, cids in profile_to_ids.items():
        if len(cids) > 1:
            clone_ids.update(cids)
    logger.info(f"Registered {len(clone_ids)} cloned candidates.")
    
    # Clear memory from first pass structures
    del profile_to_ids
    gc.collect()
    
    # -----------------------------------------------------------------
    # Step 2: Stream Candidates & Extract Heuristic Features
    # -----------------------------------------------------------------
    logger.info("Second Pass: Extracting features and honeypots...")
    
    candidate_ids = []
    candidate_store = {}
    
    # Count total candidates first to pre-allocate features array
    # Since streaming, we can iterate, but we know dataset is 100,000.
    # To be dynamic, we'll determine the size dynamically or pre-allocate to 100K.
    # Let's count them or use a dynamic list and convert to numpy array.
    # A list of small floats for 100K is tiny.
    features_list = []
    
    for candidate in stream_candidates(Config.RAW_DATA_PATH):
        cid = candidate.get("candidate_id")
        candidate_ids.append(cid)
        
        # Honeypot detection
        is_hp = is_honeypot(candidate, Config.TODAY, clone_ids)
        
        # Compute heuristic columns
        skill_match = compute_skill_match_score(candidate)
        skill_gap = compute_skill_gap_coverage(candidate)
        career_score = compute_career_score(candidate)
        exp_fit = compute_experience_fit(candidate)
        loc_score = compute_location_score(candidate)
        edu_score = compute_education_score(candidate)
        eng_score = compute_engagement_score(candidate)
        
        # Layout features (Col 0 will be RRF)
        feats = [
            0.0,            # Col 0: RRF (placeholder)
            skill_match,     # Col 1
            skill_gap,       # Col 2
            career_score,    # Col 3
            exp_fit,         # Col 4
            loc_score,       # Col 5
            edu_score,       # Col 6
            eng_score,       # Col 7
            1.0 if is_hp else 0.0  # Col 8: Honeypot flag
        ]
        features_list.append(feats)
        
        # Populate candidate store subset
        ch_store = []
        for job in candidate.get("career_history", []) or []:
            if job:
                ch_store.append({
                    "company": job.get("company"),
                    "title": job.get("title"),
                    "duration_months": job.get("duration_months"),
                    "is_current": job.get("is_current"),
                    "description": job.get("description")
                })
                
        candidate_store[cid] = {
            "current_title": candidate.get("profile", {}).get("current_title") or "",
            "current_company": candidate.get("profile", {}).get("current_company") or "",
            "years_of_experience": candidate.get("profile", {}).get("years_of_experience") or 0.0,
            "location": candidate.get("profile", {}).get("location") or "",
            "country": candidate.get("profile", {}).get("country") or "",
            "skills": candidate.get("skills", []) or [],
            "education": candidate.get("education", []) or [],
            "career_history": ch_store,
            "redrob_signals": candidate.get("redrob_signals", {}) or {}
        }
        
    features = np.array(features_list, dtype=np.float32)
    logger.info(f"Features matrix initialized with shape: {features.shape}")
    
    # -----------------------------------------------------------------
    # Step 3: Dense Embeddings (SPEC-5.5)
    # -----------------------------------------------------------------
    logger.info("Computing dense embeddings similarity...")
    
    # Load hypothetical resume
    with open("hypothetical_resume_selected.txt", "r", encoding="utf-8") as f:
        jd_text = f.read().strip()
        
    embedder = Embedder()
    jd_vector = embedder.embed_texts([jd_text])[0] # Shape: (384,)
    
    candidate_texts = []
    for candidate in stream_candidates(Config.RAW_DATA_PATH):
        candidate_texts.append(build_candidate_text(candidate))
        
    candidate_embeddings = embedder.embed_texts(candidate_texts) # (N, 384)
    
    # For legacy/test compatibility, compute unnormalized dot product first
    dense_sim_score_legacy = np.dot(candidate_embeddings, jd_vector)
    
    # Now normalize candidate embeddings and jd_vector for correct V3 cosine similarity
    candidate_embeddings_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    jd_vector_norm = jd_vector / np.linalg.norm(jd_vector)
    
    # Cosine similarities
    dense_raw = np.dot(candidate_embeddings_norm, jd_vector_norm)
    
    # Exact Rescaling formula
    # > ⚠️ **AUDIT FIX (Issue 6):** Use the exact rescaling formula below, with the comment block
    # Final Composite = 0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
    dense_scores = (dense_raw + 1.0) / 2.0
    
    logger.info(f"Dense rescaled scores range: min={np.min(dense_scores):.4f}, max={np.max(dense_scores):.4f}")
    
    # Clean raw candidate texts to free memory
    del candidate_texts
    gc.collect()
    
    # -----------------------------------------------------------------
    # Step 4: Memory-Safe BM25 Index & Scoring (SPEC-8)
    # -----------------------------------------------------------------
    logger.info("Building pruned BM25 index...")
    
    # Generator pipeline to compute document frequencies
    doc_freq = compute_doc_frequencies(Config.RAW_DATA_PATH)
    logger.info(f"Vocabulary size before pruning: {len(doc_freq)}")
    
    # Prune tokens: retain tokens with doc frequency >= 2
    pruned_vocab = {token for token, freq in doc_freq.items() if freq >= 2}
    if not pruned_vocab:
        # Fallback to keep all tokens if pruning results in an empty vocabulary (e.g. in tests)
        pruned_vocab = set(doc_freq.keys())
    logger.info(f"Vocabulary size after pruning (doc_freq >= 2): {len(pruned_vocab)}")
    
    # Build pruned corpus
    pruned_corpus = []
    for tokens in token_generator(Config.RAW_DATA_PATH):
        pruned_tokens = [t for t in tokens if t in pruned_vocab]
        pruned_corpus.append(pruned_tokens)
        
    # Free doc frequency counter before BM25 initialization
    del doc_freq
    gc.collect()
    
    # Initialize BM25
    bm25 = BM25Okapi(pruned_corpus)
    
    # Build JD Query Tokens dynamically from JD Taxonomy keys
    JD_QUERY_TOKENS = []
    for skill_name in JD_TAXONOMY.keys():
        JD_QUERY_TOKENS.extend(simple_tokenize(skill_name))
    JD_QUERY_TOKENS = list(set(JD_QUERY_TOKENS))
    logger.info(f"BM25 Query has {len(JD_QUERY_TOKENS)} unique tokens.")
    
    # Compute BM25 scores
    bm25_raw = bm25.get_scores(JD_QUERY_TOKENS)
    
    # Normalize BM25
    bm25_min = np.min(bm25_raw)
    bm25_max = np.max(bm25_raw)
    bm25_range = bm25_max - bm25_min
    if bm25_range == 0:
        bm25_range = 1.0
    bm25_scores_norm = (bm25_raw - bm25_min) / bm25_range
    
    # Clean BM25 corpus & model structures to free memory
    del pruned_corpus
    gc.collect()
    
    # -----------------------------------------------------------------
    # Step 5: Career Keyword Scoring
    # -----------------------------------------------------------------
    logger.info("Computing career keyword scores...")
    career_keyword_raw = compute_career_keyword_scores(Config.RAW_DATA_PATH)
    ck_min = np.min(career_keyword_raw)
    ck_max = np.max(career_keyword_raw)
    ck_range = ck_max - ck_min
    if ck_range == 0:
        ck_range = 1.0
    career_keyword_norm = (career_keyword_raw - ck_min) / ck_range
    
    # -----------------------------------------------------------------
    # Step 6: Reciprocal Rank Fusion (SPEC-5.6)
    # -----------------------------------------------------------------
    logger.info("Fusing ranks via Reciprocal Rank Fusion (RRF)...")
    
    def get_ranks(scores):
        # Sort descending and retrieve ranks deterministically
        # argsort with stable sort guarantees tie-breaking consistency
        idx = np.argsort(-scores, kind='stable')
        ranks = np.empty_like(idx)
        ranks[idx] = np.arange(1, len(scores) + 1)
        return ranks
        
    r_bm25 = get_ranks(bm25_scores_norm)
    r_dense = get_ranks(dense_scores)
    r_keyword = get_ranks(career_keyword_norm)
    
    k = 60.0
    rrf_raw = 1.0 / (k + r_bm25) + 1.0 / (k + r_dense) + 1.0 / (k + r_keyword)
    
    rrf_min = np.min(rrf_raw)
    rrf_max = np.max(rrf_raw)
    rrf_range = rrf_max - rrf_min
    if rrf_range == 0:
        rrf_range = 1.0
    rrf_normalized = (rrf_raw - rrf_min) / rrf_range
    
    features[:, 0] = rrf_normalized
    
    # -----------------------------------------------------------------
    # Step 7: Save SPEC-11 Artifacts
    # -----------------------------------------------------------------
    logger.info("Saving artifacts to disk...")
    
    # 1. features.npy
    np.save(os.path.join(Config.ARTIFACTS_DIR, "features.npy"), features)
    
    # 2. candidate_embeddings.npy
    np.save(os.path.join(Config.ARTIFACTS_DIR, "candidate_embeddings.npy"), candidate_embeddings)
    
    # 3. candidate_ids.pkl
    with open(os.path.join(Config.ARTIFACTS_DIR, "candidate_ids.pkl"), "wb") as f:
        pickle.dump(candidate_ids, f)
        
    # 4. bm25_model.pkl
    with open(os.path.join(Config.ARTIFACTS_DIR, "bm25_model.pkl"), "wb") as f:
        pickle.dump(bm25, f)
        
    # 5. candidate_store.pkl
    with open(os.path.join(Config.ARTIFACTS_DIR, "candidate_store.pkl"), "wb") as f:
        pickle.dump(candidate_store, f)
        
    # Also save fallback parquet file for pipeline compatibility
    df_parquet = pd.DataFrame(features, columns=[
        "rrf_score", "skill_match_score", "skill_gap_coverage", "career_score",
        "experience_fit", "location_score", "education_score", "engagement_score", "is_honeypot"
    ])
    df_parquet.insert(0, "candidate_id", candidate_ids)
    df_parquet["is_honeypot"] = df_parquet["is_honeypot"].astype(bool)
    
    # Add legacy columns for V1/test compatibility
    df_parquet["bm25_score"] = bm25_raw
    df_parquet["dense_sim_score"] = dense_sim_score_legacy
    
    # Save Parquet
    df_parquet.to_parquet(Config.FEATURES_TABLE_PATH, index=False)
    
    elapsed = time.time() - start_time
    logger.info(f"=== Precomputation Phase Complete in {elapsed:.2f}s ===")
    logger.info(f"Final features shape: {features.shape}")

if __name__ == "__main__":
    run_precompute()