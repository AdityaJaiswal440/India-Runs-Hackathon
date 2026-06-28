"""Tests for offline Redrob ranker v5."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from challenge.honeypot import honeypot_risk, is_structural_honeypot
from challenge.jd_config import GENERAL_ML_SKILLS
from challenge.redrob_ranker import _skill_bucket, rank_candidates, score_candidate
from challenge.text_match import has_trailing_partial_word_ellipsis, phrase_in_text

ROOT = Path(__file__).resolve().parents[1]
from challenge.data_paths import challenge_file

SAMPLE = challenge_file("sample_candidates.json")
CANDIDATES = challenge_file("candidates.jsonl")


def test_truncate_at_word_boundary():
    from challenge.text_match import truncate_at_word_boundary

    text = "Designed and shipped multiple ranking models for discovery feed"
    out = truncate_at_word_boundary(text, 40)
    assert not out.startswith("…")
    assert out.endswith("…")
    assert " " not in out[-5:] or out[-5:].startswith("…")


def test_snippet_avoids_partial_word_ellipsis():
    from challenge.features import _snippet_around_match

    desc = (
        "Owned the ranking layer for an e-commerce search product, evolving from "
        "keyword-based to embedding-based search across a 30M+ candidate corpus."
    )
    m = __import__("re").compile(r"ranking").search(desc)
    assert m is not None
    snippet = _snippet_around_match(desc, m, limit=72)
    assert "evolvin…" not in snippet
    assert snippet.endswith("…")


def test_substring_bugs_fixed():
    assert not phrase_in_text("map", "roadmap design")
    assert not phrase_in_text("search", "research scientist")
    assert phrase_in_text("python", "strong python skills")


def test_honeypots_not_in_top_sample():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    rows = [score_candidate(raw) for raw in data]
    rows.sort(key=lambda x: (-x.raw_score, x.candidate_id))
    for row in rows[:10]:
        assert row.components.get("honeypot_risk", 0) < 0.55


def test_impossible_tenure_demoted():
    if not CANDIDATES.exists():
        return
    target = None
    for line in CANDIDATES.open(encoding="utf-8"):
        r = json.loads(line)
        if r["candidate_id"] == "CAND_0039754":
            target = r
            break
    assert target is not None
    sc = score_candidate(target)
    assert honeypot_risk(target) >= 0.55
    assert is_structural_honeypot(target)
    top = rank_candidates(CANDIDATES, top_k=100)
    ids = {t.candidate_id for t in top}
    assert "CAND_0039754" not in ids


def test_skill_hyphen_normalization():
    assert _skill_bucket("Hugging-Face") == "secondary"
    assert _skill_bucket("fine-tuning") == "secondary"


def test_python_tensorflow_not_core_ir_names():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    row = score_candidate(data[0])
    for name in row.core_skill_names:
        assert name.lower() not in {s.lower() for s in GENERAL_ML_SKILLS}


def test_reasoning_no_midword_truncation():
    """Stage-4 snippets must not start/end with partial tokens (e.g. 'evolvin…')."""
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=20)
    bad_prefix = re.compile(r"…[a-z]{1,3}\s", re.I)
    for row in top:
        for part in row.reasoning.split("Career note:"):
            snippet = part.strip()
            if snippet.startswith("…") and bad_prefix.match(snippet[:8]):
                raise AssertionError(
                    f"mid-word truncation in {row.candidate_id}: {snippet[:80]!r}"
                )
            if has_trailing_partial_word_ellipsis(snippet):
                raise AssertionError(
                    f"trailing partial word in {row.candidate_id}: {snippet[-40:]!r}"
                )


def test_reasoning_phrase_diversity():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    texts = [r.reasoning for r in top]
    banned = "career history describes shipped retrieval/ranking work in plain language"
    assert sum(1 for t in texts if banned in t) == 0
    assert len(set(texts)) == len(texts)


def test_reasoning_uses_model_score():
    if not SAMPLE.exists():
        return
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    tmp = ROOT / "data" / "_test_sample.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in data[:50]:
            f.write(json.dumps(row) + "\n")
    top = rank_candidates(tmp, top_k=5)
    tmp.unlink(missing_ok=True)
    assert re.search(
        r"(model|calibrated)\s+\d+\.\d{4}|score\s+\d+\.\d{4}",
        top[0].reasoning.lower(),
    )
    assert len({r.reasoning for r in top}) >= 3


def test_full_dataset_structural_honeypots_below_threshold():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    traps = sum(1 for r in top if r.components.get("honeypot_risk", 0) >= 0.55)
    assert traps == 0, f"{traps} structural honeypots in top 100"


def test_scores_not_saturated():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    raw_unique = len({round(r.raw_score, 6) for r in top})
    assert raw_unique >= 20, f"only {raw_unique} unique raw scores in top 100"


def test_availability_component_present():
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    row = score_candidate(data[0])
    assert "availability" in row.components
    assert "career_semantic" in row.components


def test_calibrated_scores_monotonic():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    scores = [r.score for r in top]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert scores[0] > scores[-1]


def test_cv_profiles_not_in_top_100():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    for row in top:
        reasoning = row.reasoning.lower()
        assert "computer vision" not in reasoning or row.components.get("cv_penalty", 1.0) >= 0.9
        assert "image moderation" not in reasoning
        title = reasoning.split(",")[0].lower()
        if "junior" in title and row.components.get("cv_penalty", 1.0) < 0.8:
            raise AssertionError(f"junior CV profile at rank with cv_penalty: {row.candidate_id}")


def test_applied_scientist_not_research_only_penalty():
    """Applied scientist is strong-title only; not in RESEARCH_ONLY_TITLES."""
    from challenge.jd_config import RESEARCH_ONLY_TITLES, STRONG_TITLES
    from challenge.redrob_ranker import _research_penalty
    from challenge.features import build_index

    assert "applied scientist" in STRONG_TITLES
    assert "applied scientist" not in RESEARCH_ONLY_TITLES
    assert "research engineer" not in STRONG_TITLES

    raw = {
        "candidate_id": "TEST_APPLIED",
        "profile": {
            "current_title": "Applied Scientist",
            "years_of_experience": 6.0,
        },
        "career_history": [
            {
                "title": "Applied Scientist",
                "company": "CRED",
                "description": "Shipped hybrid retrieval and ranking in production for 2M users.",
            }
        ],
        "skills": [{"name": "Information Retrieval", "proficiency": "expert"}],
        "redrob_signals": {},
    }
    idx = build_index(raw["profile"], raw["career_history"])
    assert _research_penalty(raw["profile"], raw["career_history"], 0.7, idx) == 1.0


def test_calibrated_scores_have_floor():
    from challenge.redrob_ranker import _calibrate_scores

    raw = [1.0, 1.0, 0.5, 0.5, 0.1]
    cal = _calibrate_scores(raw)
    assert all(s >= 0.20 for s in cal)
    assert all(cal[i] >= cal[i + 1] for i in range(len(cal) - 1))


def test_proxy_relevance_excludes_ranker_signals():
    from challenge.eval_harness import proxy_relevance

    base = json.loads(SAMPLE.read_text(encoding="utf-8"))[0]
    twin = json.loads(json.dumps(base))
    twin["redrob_signals"]["recruiter_response_rate"] = 0.99
    twin["redrob_signals"]["open_to_work_flag"] = not base["redrob_signals"].get(
        "open_to_work_flag", False
    )
    assert proxy_relevance(base) == proxy_relevance(twin)


def test_behavioral_proxy_excludes_ranker_signals():
    from challenge.eval_harness import behavioral_proxy_relevance

    base = json.loads(SAMPLE.read_text(encoding="utf-8"))[0]
    twin = json.loads(json.dumps(base))
    twin["redrob_signals"]["saved_by_recruiters_30d"] = 99
    twin["redrob_signals"]["recruiter_response_rate"] = 0.99
    twin["redrob_signals"]["applications_submitted_30d"] = 50
    assert behavioral_proxy_relevance(base) == behavioral_proxy_relevance(twin)


def test_eval_harness_produces_metrics():
    from challenge.eval_harness import proxy_relevance, run_holdout_eval

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert 0.0 <= proxy_relevance(sample[0]) <= 1.0
    tmp = ROOT / "data" / "_eval_sample.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")
    report = run_holdout_eval(tmp, sample_rate=1.0, max_n=50, top_k=20)
    tmp.unlink(missing_ok=True)
    assert report.get("label_method") == "independent_jd_rubric"
    m = report["metrics_self_consistency_proxy"]
    assert "ndcg_10" in m
    assert "hidden" in report.get("note", "").lower()
    abl = report.get("weight_ablation_on_behavioral_proxy") or report.get("weight_ablation", {})
    assert abl["current"]["ndcg_10"] >= 0


def test_template_blurb_penalty_demotes_clone():
    from challenge.career_blurb import build_career_blurb_counts, template_blurb_modifier

    shared = (
        "Owned the ranking layer for an e-commerce search product, evolving it from "
        "keyword-based to embedding-based search across a 30M+ candidate corpus."
    )
    counts = build_career_blurb_counts(
        [
            {
                "candidate_id": f"CAND_{i:07d}",
                "career_history": [{"description": shared}],
            }
            for i in range(120)
        ]
    )
    assert counts
    assert template_blurb_modifier([{"description": shared}], counts) < 0.9


def test_not_open_to_work_not_in_top10():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    need = {r.candidate_id for r in top[:10]}
    otw: dict[str, bool] = {}
    with CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            cid = raw["candidate_id"]
            if cid in need:
                otw[cid] = bool(raw["redrob_signals"].get("open_to_work_flag", False))
            if len(otw) == len(need):
                break
    for cid in need:
        assert otw.get(cid) is True, f"{cid} not open to work in top-10"


def test_reasoning_compact_and_unique():
    if not CANDIDATES.exists():
        return
    top = rank_candidates(CANDIDATES, top_k=100)
    texts = [r.reasoning for r in top]
    assert len(set(texts)) == len(texts)
    for row in top:
        assert len(row.reasoning) <= 420, f"{row.candidate_id} reasoning too long"
        assert re.search(r"score\s+\d+\.\d{4}|rank\s+\d+", row.reasoning.lower())


def test_behavioral_twin_breaks_tie():
    base = json.loads(SAMPLE.read_text(encoding="utf-8"))[0]
    twin_a = json.loads(json.dumps(base))
    twin_b = json.loads(json.dumps(base))
    twin_a["candidate_id"] = "TWIN_A"
    twin_b["candidate_id"] = "TWIN_B"
    twin_a["redrob_signals"]["recruiter_response_rate"] = 0.85
    twin_a["redrob_signals"]["saved_by_recruiters_30d"] = 6
    twin_a["redrob_signals"]["open_to_work_flag"] = True
    twin_b["redrob_signals"]["recruiter_response_rate"] = 0.12
    twin_b["redrob_signals"]["saved_by_recruiters_30d"] = 0
    twin_b["redrob_signals"]["open_to_work_flag"] = False
    sa = score_candidate(twin_a)
    sb = score_candidate(twin_b)
    assert sa.raw_score > sb.raw_score


def test_keyword_stuffer_far_from_top():
    from challenge.jd_config import CORE_SKILL_PHRASES

    skills = [
        {
            "name": p.title(),
            "proficiency": "expert",
            "endorsements": 40,
            "duration_months": 0,
        }
        for p in CORE_SKILL_PHRASES[:20]
    ]
    stuffer = {
        "candidate_id": "STUFFER_TEST",
        "profile": {
            "current_title": "Marketing Manager",
            "years_of_experience": 5.0,
            "summary": "Enterprise sales and brand campaigns.",
            "headline": "Marketing Manager",
            "location": "Mumbai",
            "country": "India",
        },
        "career_history": [
            {
                "title": "Marketing Manager",
                "company": "Acme",
                "description": "Led brand campaigns and social media growth.",
            }
        ],
        "skills": skills,
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.5},
    }
    sc = score_candidate(stuffer)
    assert sc.raw_score < 0.25


def test_honeypot_recall_scan_runs():
    if not CANDIDATES.exists():
        return
    from challenge.honeypot import honeypot_risk, is_structural_honeypot
    from challenge.redrob_ranker import load_candidates

    n = flagged = 0
    for raw in load_candidates(CANDIDATES):
        n += 1
        if is_structural_honeypot(raw) or honeypot_risk(raw) >= 0.55:
            flagged += 1
        if n >= 20000:
            break
    assert n > 0
    assert flagged >= 5, f"honeypot recall suspiciously low: {flagged}/{n}"


def test_runtime_under_budget():
    """Full 100K ranking must finish under 300s (Stage-3 hard limit)."""
    if not CANDIDATES.exists():
        return
    t0 = time.perf_counter()
    rank_candidates(CANDIDATES, top_k=100)
    elapsed = time.perf_counter() - t0
    assert elapsed < 300.0, f"ranking took {elapsed:.1f}s — exceeds 300s budget"
    assert elapsed < 180.0, f"ranking took {elapsed:.1f}s — target <180s with rerank"


def test_no_embeddings_top10_overlap():
    """Fresh clone without npy must stay within 1 ID of artifact path on top-10."""
    if not CANDIDATES.exists():
        return
    import challenge.embeddings as emb_mod

    emb_top = rank_candidates(CANDIDATES, top_k=100)
    emb_ids = [r.candidate_id for r in emb_top]

    import os

    empty = ROOT / "data" / "_no_emb_dir"
    empty.mkdir(exist_ok=True)
    old_root = emb_mod._DEFAULT_DIR
    old_req = os.environ.get("RANKER_REQUIRE_EMBEDDINGS")
    emb_mod._DEFAULT_DIR = empty
    import challenge.redrob_ranker as rr

    rr._EMBED_STORE = None
    os.environ["RANKER_REQUIRE_EMBEDDINGS"] = "0"
    try:
        no_top = rank_candidates(CANDIDATES, top_k=100)
    finally:
        emb_mod._DEFAULT_DIR = old_root
        rr._EMBED_STORE = None
        if old_req is None:
            os.environ.pop("RANKER_REQUIRE_EMBEDDINGS", None)
        else:
            os.environ["RANKER_REQUIRE_EMBEDDINGS"] = old_req

    no_ids = [r.candidate_id for r in no_top]
    overlap10 = len(set(emb_ids[:10]) & set(no_ids[:10]))
    assert overlap10 >= 9, f"no-embeddings top-10 set overlap {overlap10}/10"
    overlap5 = len(set(emb_ids[:5]) & set(no_ids[:5]))
    assert overlap5 >= 4, f"no-embeddings top-5 set overlap {overlap5}/5"


def test_embeddings_guard_blocks_silent_fallback():
    import os
    import tempfile

    import challenge.embeddings as emb_mod
    import challenge.redrob_ranker as rr
    from challenge.embeddings import guard_canonical_embeddings

    empty = Path(tempfile.mkdtemp())
    old_dir = emb_mod._DEFAULT_DIR
    old_store = rr._EMBED_STORE
    old_req = os.environ.get("RANKER_REQUIRE_EMBEDDINGS")
    try:
        emb_mod._DEFAULT_DIR = empty
        rr._EMBED_STORE = None
        os.environ["RANKER_REQUIRE_EMBEDDINGS"] = "1"
        try:
            guard_canonical_embeddings(rr._embedding_store())
            raise AssertionError("expected RuntimeError for missing embeddings")
        except RuntimeError as exc:
            assert "Committed embeddings not loaded" in str(exc)
    finally:
        emb_mod._DEFAULT_DIR = old_dir
        rr._EMBED_STORE = old_store
        if old_req is None:
            os.environ.pop("RANKER_REQUIRE_EMBEDDINGS", None)
        else:
            os.environ["RANKER_REQUIRE_EMBEDDINGS"] = old_req


def test_cross_encoder_disabled_by_default():
    import os

    from challenge.rerank import cross_encoder_enabled

    assert os.environ.get("RANKER_USE_CROSS_ENCODER", "0") in ("0", "")
    assert cross_encoder_enabled() is False


def test_submission_artifact_matches_ranker():
    """Committed era.csv must be byte-identical to fresh rank.py output."""
    artifact = ROOT / "era.csv"
    if not artifact.exists() or not CANDIDATES.exists():
        return
    import hashlib
    import os
    import tempfile

    from challenge.redrob_ranker import write_submission

    os.environ["RANKER_USE_CROSS_ENCODER"] = "0"
    fresh = rank_candidates(CANDIDATES, top_k=100)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = Path(tmp.name)
    write_submission(fresh, out)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == hashlib.sha256(
        out.read_bytes()
    ).hexdigest(), "era.csv stale vs canonical ranker (CE off)"
    out.unlink(missing_ok=True)