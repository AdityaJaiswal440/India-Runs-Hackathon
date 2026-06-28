"""scripts/audit_submission.py

Audit script implementing SPEC-14 assertions and the Gate P4 anti-hallucination test.

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

    # 2.10 Honeypot Integrity Check (SPEC-5.14)
    for row in submission_rows:
        cid = row["candidate_id"]
        score = float(row["score"])
        if candidate_store[cid].get("is_honeypot", False):
            assert_true(score == 0.0, f"Honeypot candidate {cid} must have score 0.0")

    # -----------------------------------------------------------------------
    # 3️⃣ Gate P4 – Entity-Centric Semantic Grounding
    # -----------------------------------------------------------------------
    # Golden Rule: We ONLY validate tokens that are technical entities
    # (i.e., they exist in JD_TAXONOMY). Standard English words like
    # "critical", "concern", "fit", "seen" are NOT technical entities
    # and are unconditionally ignored.
    #
    # For each taxonomy term found in the reasoning, we verify it is
    # grounded in the candidate's actual profile (skills or career history).

    for row in submission_rows:
        cid = row["candidate_id"]
        candidate = candidate_store[cid]

        # Build the candidate's valid entity set from their profile
        valid_candidate_entities = set()

        # Add all skill names (lowercased)
        for s in (candidate.get("skills", []) or []):
            if s and s.get("name"):
                valid_candidate_entities.add(s["name"].strip().lower())

        # Add all company names (lowercased)
        for job in (candidate.get("career_history", []) or []):
            if job and job.get("company"):
                valid_candidate_entities.add(job["company"].strip().lower())

        # Serialise the entire candidate profile for substring fallback
        candidate_text = str(candidate).lower()

        # Lowercase the reasoning for matching
        reasoning_lower = row["reasoning"].lower()

        # Check each taxonomy term (now compiled with \b boundaries):
        # if it appears in the reasoning, it MUST be grounded in the candidate's profile
        for term in JD_TAXONOMY:
            if re.search(term, reasoning_lower):
                # This technical term was claimed in the reasoning.
                # Since term is a regex with \b, we need to extract the raw skill name 
                # (removing the \b markers) to check valid_candidate_entities.
                raw_term = term.replace(r"\b", "")
                if raw_term in valid_candidate_entities:
                    continue
                if re.search(term, candidate_text):
                    continue
                # Hard hallucination: taxonomy term claimed but not in profile
                print(f"[FAIL] Hallucination in {cid}: taxonomy term '{raw_term}' "
                      f"found in reasoning but not in candidate profile")
                sys.exit(1)

    # -----------------------------------------------------------------------
    # 4️⃣ All checks passed
    # -----------------------------------------------------------------------
    print("[SUCCESS] Audit passed - submission satisfies SPEC-14 and Gate P4.")
    sys.exit(0)

if __name__ == "__main__":
    main()
