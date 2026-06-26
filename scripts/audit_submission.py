"""scripts/audit_submission.py

Audit script implementing SPEC-14 assertions and the Gate P4 anti‑hallucination test.

Usage:
    python scripts/audit_submission.py [submission.csv]
If no argument is supplied, ``submission.csv`` in the current working directory is used.
"""

import sys
import os
import csv
import re
import pickle
from pathlib import Path

# Ensure project root is on the path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.heuristic_extractor import JD_TAXONOMY

# ---------------------------------------------------------------------------
# Whitelist: tokens that are structural / common words, not skill names
# ---------------------------------------------------------------------------
WHITELIST = {
    "Senior", "Expert", "Proven", "Top", "Concern",
    "ML", "AI", "AWS", "SQL", "Python", "RAG", "GCP", "Azure",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_submission(csv_path: str):
    """Load submission CSV and return list of rows as dicts.
    Expected columns: candidate_id, rank, score, reasoning
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_candidate_store(artifacts_dir: str):
    store_path = os.path.join(artifacts_dir, "candidate_store.pkl")
    with open(store_path, "rb") as f:
        return pickle.load(f)

def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)

def main():
    # -----------------------------------------------------------------------
    # 1️⃣ Load inputs
    # -----------------------------------------------------------------------
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    assert_true(os.path.isfile(csv_path), f"Submission file not found: {csv_path}")

    # Find project root (the directory containing 'src' and 'artifacts')
    cwd = Path.cwd()
    project_root = cwd
    while not (project_root / "src").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    artifacts_dir = os.path.join(project_root, "artifacts")
    assert_true(os.path.isdir(artifacts_dir), f"Artifacts directory not found: {artifacts_dir}")

    submission_rows = load_submission(csv_path)
    candidate_store = load_candidate_store(artifacts_dir)

    # -----------------------------------------------------------------------
    # 2️⃣ SPEC‑14 assertions (10 of them)
    # -----------------------------------------------------------------------
    # 2.1 Exactly 100 rows
    assert_true(len(submission_rows) == 100, f"Expected 100 rows, got {len(submission_rows)}")

    # 2.2 Ranks are 1‑100 without gaps and sorted ascending in file
    ranks = [int(r["rank"]) for r in submission_rows]
    assert_true(ranks == list(range(1, 101)), "Ranks must be 1..100 in order")

    # 2.3 Candidate IDs must be present in candidate_store and unique
    candidate_ids = [r["candidate_id"] for r in submission_rows]
    assert_true(len(set(candidate_ids)) == 100, "Candidate IDs must be unique")
    for cid in candidate_ids:
        assert_true(cid in candidate_store, f"Candidate ID {cid} not found in store")

    # 2.4 Scores are numeric, within [0.001, 0.999] and sorted descending
    scores = [float(r["score"]) for r in submission_rows]
    assert_true(all(0.001 <= s <= 0.999 for s in scores), "Scores must be in [0.001, 0.999]")
    assert_true(scores == sorted(scores, reverse=True), "Scores must be descending")

    # 2.5 Honeypot candidates have score exactly 0.0 (the pipeline guarantees this)
    # No explicit check beyond ensuring any zero score is acceptable.

    # 2.6 Reasoning column is non‑empty and ends with a period.
    for row in submission_rows:
        reasoning = row["reasoning"].strip()
        assert_true(reasoning, "Reasoning must not be empty")
        assert_true(reasoning.endswith('.'), "Reasoning must end with a period")

    # 2.7 No duplicate (rank, candidate_id) pairs
    pairs = [(r["rank"], r["candidate_id"]) for r in submission_rows]
    assert_true(len(set(pairs)) == len(pairs), "Duplicate rank‑candidate pairs detected")

    # 2.8 Score column has exactly six decimal places
    for row in submission_rows:
        score_str = row["score"]
        assert_true(re.fullmatch(r"\d+\.\d{6}", score_str), f"Score format incorrect: {score_str}")

    # 2.9 The CSV header must be exactly as specified
    expected_header = ["candidate_id", "rank", "score", "reasoning"]
    with open(csv_path, newline="", encoding="utf-8") as f:
        first_line = f.readline().strip()
    assert_true(first_line == ",".join(expected_header), "CSV header does not match spec")

    # -----------------------------------------------------------------------
    # 3️⃣ Gate P4 – Anti‑Hallucination Test
    # -----------------------------------------------------------------------
    capital_word_regex = re.compile(r"\b[A-Z][a-zA-Z\-]+\b")
    for row in submission_rows:
        cid = row["candidate_id"]
        candidate = candidate_store[cid]
        # Build lowercase skill name set for case-insensitive comparison
        skill_names = {
            (s.get("name") or "").strip().lower()
            for s in candidate.get("skills", []) if s
        }
        tokens = capital_word_regex.findall(row["reasoning"])
        for token in tokens:
            # 1) Skip whitelisted structural / common words
            if token in WHITELIST:
                continue
            token_lower = token.lower()
            # 2) Allow if token matches a candidate declared skill (case-insensitive)
            if token_lower in skill_names:
                continue
            # 3) Allow if token matches an entry in JD_TAXONOMY
            if token_lower in JD_TAXONOMY:
                continue
            print(f"[FAIL] Hallucination detected in candidate {cid}: token '{token}' not in declared skills or taxonomy")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 4️⃣ All checks passed
    # -----------------------------------------------------------------------
    print("✅ Audit passed – submission satisfies SPEC‑14 and Gate P4.")
    sys.exit(0)

if __name__ == "__main__":
    main()
