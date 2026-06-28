# scripts/run_sensitivity_test.py
import os
import sys
import csv
import json
import pickle
import random
import copy
import numpy as np
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import Config
from src.features.heuristic_extractor import behavioral_multiplier

def load_top_100_ids():
    csv_path = "submission.csv"
    if not os.path.exists(csv_path):
        return set()
    ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["candidate_id"])
    return ids

def main():
    print("=== STARTING RANKING PIPELINE SENSITIVITY TEST ===")
    
    # 1. Setup Sandbox Directory
    sandbox_dir = "tests/sensitivity_test"
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # Paths for original artifacts
    orig_features_path = os.path.join("artifacts", "features.npy")
    orig_ids_path = os.path.join("artifacts", "candidate_ids.pkl")
    orig_store_path = os.path.join("artifacts", "candidate_store.pkl")
    orig_today_path = os.path.join("artifacts", "dataset_today.txt")
    
    # Check if original artifacts exist
    if not (os.path.exists(orig_features_path) and os.path.exists(orig_ids_path) and os.path.exists(orig_store_path)):
        print("[FAIL] Production artifacts not found in artifacts/!")
        sys.exit(1)
        
    # Copy dataset_today.txt to sandbox
    if os.path.exists(orig_today_path):
        with open(orig_today_path, "r", encoding="utf-8") as f:
            today_str = f.read().strip()
        with open(os.path.join(sandbox_dir, "dataset_today.txt"), "w", encoding="utf-8") as f:
            f.write(today_str)
            
    # Load original candidate store
    with open(orig_store_path, "rb") as f:
        candidate_store = pickle.load(f)
        
    # Load original candidate IDs
    with open(orig_ids_path, "rb") as f:
        candidate_ids = pickle.load(f)
        
    # Load original features matrix
    features = np.load(orig_features_path)
    
    print(f"Loaded original candidate store with {len(candidate_ids)} entries.")
    
    # 2. Select one truly random candidate not currently in top 100
    top_100_ids = load_top_100_ids()
    eligible_ids = [cid for cid in candidate_ids if cid not in top_100_ids]
    
    if not eligible_ids:
        print("[FAIL] No eligible candidates found outside the top 100!")
        sys.exit(1)
        
    random_cid = random.choice(eligible_ids)
    random_idx = candidate_ids.index(random_cid)
    print(f"Selected random candidate: {random_cid} (at index {random_idx})")
    
    # Load God-Tier candidate
    god_path = os.path.join("artifacts", "golden_candidate.json")
    if not os.path.exists(god_path):
        print(f"[FAIL] God candidate file not found at {god_path}!")
        sys.exit(1)
        
    with open(god_path, "r", encoding="utf-8") as f:
        god_cand = json.load(f)
        
    god_cid = god_cand["candidate_id"] # CAND_GOD_TIER_001
    
    # 3. Build Shadow Store and inject God Candidate
    shadow_store = copy.deepcopy(candidate_store)
    shadow_store[god_cid] = {
        "current_title": god_cand.get("profile", {}).get("current_title") or "",
        "current_company": god_cand.get("profile", {}).get("current_company") or "",
        "years_of_experience": god_cand.get("profile", {}).get("years_of_experience") or 0.0,
        "location": god_cand.get("profile", {}).get("location") or "",
        "country": god_cand.get("profile", {}).get("country") or "",
        "skills": god_cand.get("skills", []) or [],
        "education": god_cand.get("education", []) or [],
        "career_history": god_cand.get("career_history", []),
        "redrob_signals": god_cand.get("redrob_signals", {}) or {}
    }
    
    shadow_candidate_ids = copy.deepcopy(candidate_ids)
    shadow_candidate_ids.append(god_cid)
    
    # Append a row of 1.0s (except honeypot = 0.0) for the God Candidate
    god_feats = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    shadow_features = np.vstack([features, god_feats])
    
    # Save shadow store artifacts in sandbox
    with open(os.path.join(sandbox_dir, "candidate_store.pkl"), "wb") as f:
        pickle.dump(shadow_store, f)
    with open(os.path.join(sandbox_dir, "candidate_ids.pkl"), "wb") as f:
        pickle.dump(shadow_candidate_ids, f)
    np.save(os.path.join(sandbox_dir, "features.npy"), shadow_features)
    
    print("Shadow store saved to sandbox.")
    
    # 4. Execute Pipeline Scoring Logic (the scoring function)
    STATIC_WEIGHTS = np.array([0.30, 0.22, 0.06, 0.20, 0.10, 0.07, 0.03, 0.02, 0.00], dtype=np.float32)
    base_scores = shadow_features[:, :9] @ STATIC_WEIGHTS
    
    final_scores = np.zeros_like(base_scores)
    today = Config.TODAY
    
    for idx, cid in enumerate(shadow_candidate_ids):
        candidate = shadow_store[cid]
        mult = behavioral_multiplier(candidate, today)
        final_scores[idx] = base_scores[idx] * mult
        
    # Clamp and apply honeypot mask
    final_scores = np.clip(final_scores, 0.001, 0.999)
    honeypot_mask = shadow_features[:, 8] == 1.0
    final_scores[honeypot_mask] = 0.0
    
    # Deterministic sorting
    all_candidates = list(zip(shadow_candidate_ids, final_scores))
    all_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    # Export shadow ranking to shadow_ranking.csv
    shadow_csv_path = os.path.join(sandbox_dir, "shadow_ranking.csv")
    with open(shadow_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score) in enumerate(all_candidates, 1):
            writer.writerow([cid, rank, f"{score:.6f}", "shadow ranking score"])
            
    print(f"Shadow ranking exported to {shadow_csv_path}")
    
    # 5. Extract scores and ranks for comparison
    ranking_ids = [item[0] for item in all_candidates]
    ranking_scores = [item[1] for item in all_candidates]
    
    god_rank = ranking_ids.index(god_cid) + 1
    god_score = float(ranking_scores[ranking_ids.index(god_cid)])
    
    random_rank = ranking_ids.index(random_cid) + 1
    random_score = float(ranking_scores[ranking_ids.index(random_cid)])
    
    delta_score = god_score - random_score
    
    # 6. Report and analysis
    print("\n" + "="*50)
    print("SENSITIVITY COMPARISON REPORT")
    print("="*50)
    print(f"God-Tier Candidate ({god_cid}):")
    print(f"  Rank: #{god_rank} | Score: {god_score:.6f}")
    print(f"Random Candidate ({random_cid}):")
    print(f"  Rank: #{random_rank} | Score: {random_score:.6f}")
    print(f"Delta Score (God_Score - Random_Score): {delta_score:.6f}")
    print(f"Ranking gap: {random_rank - god_rank} positions")
    
    # Spot displacement analysis
    # Let's count how many spots the random candidate is from the top 100
    if random_rank <= 100:
        print("[ANALYSIS] The random candidate actually entered the Top 100, displacing 1 candidate.")
    else:
        print(f"[ANALYSIS] The random candidate did not enter the Top 100. It is positioned at rank #{random_rank}.")
        
    print("\n✅ SENSITIVITY TEST COMPLETE (NO PRODUCTION FILES AFFECTED)")
    print("="*50)

if __name__ == "__main__":
    main()
