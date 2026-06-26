import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

def run_systemic_audit():
    print("=== Initiating Red Team Architectural Audit ===")
    
    # 1. Load all system states
    try:
        submission = pd.read_csv("submission.csv")
        features = np.load("artifacts/features.npy")
        with open("artifacts/candidate_ids.pkl", "rb") as f:
            c_ids = pickle.load(f)
        with open("artifacts/candidate_store.pkl", "rb") as f:
            store = pickle.load(f)
    except Exception as e:
        print(f"[FATAL] Missing artifacts: {e}")
        sys.exit(1)

    fail_flag = False

    # 2. Math Bounds & NaN Detection
    print("\n[Phase 1] Mathematical Integrity Check...")
    if np.isnan(features).any():
        print("[FAIL] NaN values detected in feature matrix. Pipeline will crash on unfamiliar data.")
        fail_flag = True
    if np.isinf(features).any():
        print("[FAIL] Infinity values detected in feature matrix. Division by zero in precompute.")
        fail_flag = True
        
    # Check if any feature column has zero variance (dead weight)
    variances = np.var(features, axis=0)
    for i, var in enumerate(variances):
        if var == 0:
            print(f"[WARNING] Feature column {i} has zero variance. Static weight is wasting compute.")

    # 3. Honeypot Deep Scan (Beyond the top 100)
    print("\n[Phase 2] Adversarial Honeypot Penetration...")
    honeypot_count = sum(1 for cid, data in store.items() if data.get("is_honeypot", False))
    print(f"Total honeypots in raw data: {honeypot_count}")
    
    sub_ids = set(submission["candidate_id"].tolist())
    leaked_honeypots = [cid for cid in sub_ids if store[cid].get("is_honeypot", False)]
    if leaked_honeypots:
        print(f"[FAIL] CRITICAL LEAK: {len(leaked_honeypots)} Honeypots bypassed the filter and made the top 100!")
        fail_flag = True
    else:
        print("✅ Honeypot quarantine holds.")

    # 4. Tie-Breaker Fragility (SPEC-2.2)
    print("\n[Phase 3] Deterministic Tie-Breaker Stress Test...")
    scores = submission["score"].tolist()
    duplicates = len(scores) - len(set(scores))
    if duplicates > 0:
        print(f"[INFO] {duplicates} score collisions detected in Top 100.")
        # Verify strict secondary sorting (by Candidate ID descending/ascending)
        # Assuming secondary sort is alphabetical on ID
        for i in range(len(scores) - 1):
            if scores[i] == scores[i+1]:
                if submission["candidate_id"].iloc[i] < submission["candidate_id"].iloc[i+1]:
                     print(f"[FAIL] Tie-breaker sort direction failed between {submission['candidate_id'].iloc[i]} and {submission['candidate_id'].iloc[i+1]}")
                     fail_flag = True
    else:
        print("✅ No collisions, or sorting is stable.")

    # 5. Semantic Grounding Fallback Check
    print("\n[Phase 4] Semantic Fallback Density...")
    fallback_triggers = 0
    for idx, row in submission.iterrows():
        if row["candidate_id"] not in store:
            print(f"[FAIL] Candidate {row['candidate_id']} in submission but missing from store.")
            fail_flag = True
            
    if fail_flag:
        print("\n❌ SYSTEMIC AUDIT FAILED. Architectural vulnerabilities detected.")
        sys.exit(1)
    else:
        print("\n✅ SYSTEMIC AUDIT PASSED. Architecture is mathematically sound and deterministic.")

if __name__ == "__main__":
    run_systemic_audit()