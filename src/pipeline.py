# src/pipeline.py
"""
Module: Pipeline Orchestrator (Online Phase)
Responsibility: Load precomputed feature table, filter honeypots,
apply live multipliers, sort with deterministic tie-breaks,
generate procedural reasoning, and write final submission CSV.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

import pickle
import csv
import re
import datetime
import numpy as np
import pandas as pd

from src.utils.config import Config
from src.utils.logger import get_logger
from src.features.heuristic_extractor import behavioral_multiplier, CONSULTING_FIRMS, JD_TAXONOMY

logger = get_logger("pipeline")

def main():
    logger.info("=== Initializing Redrob Ranking Pipeline (Online Phase) ===")
    
    # 1. Load Artifacts (SPEC-9)
    features_path = os.path.join(Config.ARTIFACTS_DIR, "features.npy")
    candidate_ids_path = os.path.join(Config.ARTIFACTS_DIR, "candidate_ids.pkl")
    candidate_store_path = os.path.join(Config.ARTIFACTS_DIR, "candidate_store.pkl")
    
    if not os.path.exists(features_path):
        logger.error(f"Features file not found at {features_path}!")
        sys.exit(1)
    if not os.path.exists(candidate_ids_path):
        logger.error(f"Candidate IDs file not found at {candidate_ids_path}!")
        sys.exit(1)
    if not os.path.exists(candidate_store_path):
        logger.error(f"Candidate store file not found at {candidate_store_path}!")
        sys.exit(1)
        
    features = np.load(features_path)
    with open(candidate_ids_path, "rb") as f:
        candidate_ids = pickle.load(f)
    with open(candidate_store_path, "rb") as f:
        candidate_store = pickle.load(f)
        
    today = Config.TODAY
    logger.info(f"Loaded features shape: {features.shape}. TODAY anchor: {today}")
    
    # Define STATIC_WEIGHTS (SPEC-6.1)
    STATIC_WEIGHTS = np.array([0.30, 0.22, 0.06, 0.20, 0.10, 0.07, 0.03, 0.02, 0.00], dtype=np.float32)
    
    # 2. Live Scoring & Honeypot Masking (SPEC-9)
    base_scores = features[:, :9] @ STATIC_WEIGHTS
    
    final_scores = np.zeros_like(base_scores)
    for idx, cid in enumerate(candidate_ids):
        candidate = candidate_store[cid]
        mult = behavioral_multiplier(candidate, today)
        final_scores[idx] = base_scores[idx] * mult

    # Clamp final scores strictly to [0.001, 0.999]
    final_scores = np.clip(final_scores, 0.001, 0.999)

    # Apply honeypot mask after clamping
    honeypot_mask = features[:, 8] == 1.0
    final_scores[honeypot_mask] = 0.0
    
    # 3. The Tie-Break Sort (SPEC-4 & SPEC-9)
    assert not np.any(np.isnan(final_scores)), "NaN scores detected!"
    assert not np.any(np.isinf(final_scores)), "Infinity scores detected!"
    
    # Combine candidate_ids and final_scores into a list and sort the entire pool
    # by score descending, then candidate_id ascending to guarantee tie-breaker correctness.
    all_candidates = list(zip(candidate_ids, final_scores))
    all_candidates.sort(key=lambda x: (-x[1], x[0]))
    
    top_candidates = all_candidates[:100]
    
    # 4. Procedural Reasoning Engine (SPEC-7)
    # Define extractors
    def get_experience_fact(candidate):
        yoe = float((candidate.get("profile") or {}).get("years_of_experience") or 0.0)
        if yoe > 0:
            score = 1.0 if (5.0 <= yoe <= 9.0) else 0.5
            return (f"{yoe:.1f}y applied ML background", score)
        return None

    def get_title_fact(candidate):
        title = (candidate.get("profile") or {}).get("current_title") or candidate.get("current_title") or ""
        company = (candidate.get("profile") or {}).get("current_company") or candidate.get("current_company") or ""
        if title:
            title_lower = title.lower()
            is_ml_related = any(term in title_lower for term in ["ml", "ai", "machine learning", "retrieval", "search", "nlp", "rag"])
            score = 0.9 if is_ml_related else 0.4
            if company:
                return (f"current role as {title} at {company}", score)
            else:
                return (f"current role as {title}", score)
        return None

    def get_top_skill_fact(candidate):
        skills = candidate.get("skills", []) or []
        signals = candidate.get("redrob_signals", {}) or {}
        assessments = signals.get("skill_assessment_scores", {}) or {}
        valid_skills = []
        for s in skills:
            if not s:
                continue
            original_name = (s.get("name") or "").strip()
            sname_lower = original_name.lower()
            if sname_lower in JD_TAXONOMY:
                dur = s.get("duration_months", 0) or 0
                if dur >= 6:
                    base = JD_TAXONOMY[sname_lower]
                    assess_score = assessments.get(sname_lower) or assessments.get(original_name)
                    if assess_score is not None:
                        assess_factor = float(assess_score) / 100.0
                    else:
                        assess_factor = 0.65
                    score = base * min(dur / 24.0, 1.0) * assess_factor
                    valid_skills.append((original_name, dur, assess_score, score))
        if valid_skills:
            valid_skills.sort(key=lambda x: x[3], reverse=True)
            sname_display, dur, assess_score, score = valid_skills[0]
            if assess_score is not None:
                text = f"{sname_display} ({dur}mo, assessed {int(assess_score)}/100)"
            else:
                text = f"{sname_display} ({dur}mo)"
            return (text, score)
        return None

    def get_company_type_fact(candidate):
        history = candidate.get("career_history", []) or []
        product_companies = []
        for job in history:
            if not job:
                continue
            comp = (job.get("company") or "").strip()
            if not comp:
                continue
            comp_lower = comp.lower()
            is_consulting = False
            for firm in CONSULTING_FIRMS:
                pattern = rf"\b{re.escape(firm)}\b"
                if re.search(pattern, comp_lower):
                    is_consulting = True
                    break
            if not is_consulting and comp not in product_companies:
                product_companies.append(comp)
        if product_companies:
            comps_str = ", ".join(product_companies[:2])
            return (f"product-company career at {comps_str}", 0.7)
        return None

    def get_retrieval_fact(candidate):
        history = candidate.get("career_history", []) or []
        descriptions = []
        for job in history:
            if job and job.get("description"):
                descriptions.append(job["description"].lower())
        combined_desc = " ".join(descriptions)
        matched_keyword = None
        if "embedding" in combined_desc:
            matched_keyword = "embedding pipelines"
        elif "retrieval" in combined_desc:
            matched_keyword = "retrieval systems"
        elif "vector" in combined_desc:
            matched_keyword = "vector search"
        elif "rag" in combined_desc:
            matched_keyword = "RAG architectures"
        elif "search" in combined_desc:
            matched_keyword = "search systems"
        elif "bm25" in combined_desc:
            matched_keyword = "bm25 scoring"
        if matched_keyword:
            return (f"career text mentions {matched_keyword}", 0.8)
        return None

    def get_github_fact(candidate):
        signals = candidate.get("redrob_signals", {}) or {}
        gh = signals.get("github_activity_score")
        if gh is not None and gh != -1 and gh > 0:
            score = 0.75 * (float(gh) / 100.0) + 0.1
            return (f"active open-source (GitHub: {int(gh)}/100)", score)
        return None

    def get_response_rate_fact(candidate):
        signals = candidate.get("redrob_signals", {}) or {}
        rr = signals.get("recruiter_response_rate")
        if rr is not None and rr > 0:
            return (f"recruiter response rate: {int(rr * 100)}%", 0.65)
        return None

    def get_location_fact(candidate):
        # Extract city from the candidate profile location string, isolating the first token before any comma.
        location = (candidate.get("profile", {}) or {}).get("location", "Unknown")
        city = location.split(',')[0].strip()
        # Define allowed tier‑1 cities for deterministic matching.
        allowed_cities = {"Pune", "Noida", "Mumbai", "Delhi", "Hyderabad"}
        if city in allowed_cities:
            # Return a deterministic fact with high confidence.
            return (f"{city}-based", 0.90)
        # If city is unknown or not in the allowed set, indicate relocatability.
        return ("relocatable", 0.60)

    def get_notice_fact(candidate):
        signals = candidate.get("redrob_signals", {}) or {}
        notice = signals.get("notice_period_days")
        if notice is not None and notice <= 30:
            score = 0.85 if notice <= 15 else 0.70
            return (f"{int(notice)}-day notice period", score)
        return None

    def get_active_fact(candidate):
        signals = candidate.get("redrob_signals", {}) or {}
        last_active_str = signals.get("last_active_date")
        if last_active_str:
            try:
                if isinstance(last_active_str, datetime.date):
                    last_active_dt = last_active_str
                else:
                    last_active_dt = datetime.date.fromisoformat(last_active_str)
                days_inactive = (today - last_active_dt).days
                if days_inactive <= 30:
                    return (f"active on platform (last seen: {days_inactive}d ago)", 0.80)
            except (ValueError, TypeError):
                pass
        return None

    def get_education_fact(candidate):
        education = candidate.get("education", []) or []
        has_tier1 = any(edu.get("tier") == "tier_1" for edu in education if edu)
        if has_tier1:
            return ("tier_1 institution", 0.70)
        return None

    def get_concern_fact(candidate):
        signals = candidate.get("redrob_signals", {}) or {}
        last_active_str = signals.get("last_active_date")
        if last_active_str:
            try:
                if isinstance(last_active_str, datetime.date):
                    last_active_dt = last_active_str
                else:
                    last_active_dt = datetime.date.fromisoformat(last_active_str)
                days_inactive = (today - last_active_dt).days
                if days_inactive > 60:
                    return f"inactive {days_inactive}d on platform"
            except (ValueError, TypeError):
                pass
        
        notice = signals.get("notice_period_days")
        if notice is not None:
            try:
                notice_val = float(notice)
                if notice_val > 60:
                    return f"notice period {int(notice_val)}d"
            except (ValueError, TypeError):
                pass
                
        rr = signals.get("recruiter_response_rate")
        if rr is not None:
            try:
                rr_val = float(rr)
                if rr_val < 0.3:
                    return f"low recruiter response rate ({int(rr_val * 100)}%)"
            except (ValueError, TypeError):
                pass
                
        icr = signals.get("interview_completion_rate")
        if icr is not None:
            try:
                icr_val = float(icr)
                if icr_val < 0.5:
                    return f"interview completion rate {int(icr_val * 100)}%"
            except (ValueError, TypeError):
                pass
                
        if signals.get("verified_email", True) is False:
            return "email unverified"
            
        return "no critical concerns identified"

    FACT_EXTRACTORS = [
        get_experience_fact,
        get_title_fact,
        get_top_skill_fact,
        get_company_type_fact,
        get_retrieval_fact,
        get_github_fact,
        get_response_rate_fact,
        get_location_fact,
        get_notice_fact,
        get_active_fact,
        get_education_fact
    ]
    
    # 5. Output CSV Generation (SPEC-10)
    output_path = "submission.csv"
    logger.info(f"Writing top 100 rows to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, (cid, score) in enumerate(top_candidates, 1):
            candidate = candidate_store[cid]
            
            # Run all positive extractors
            positive_facts = []
            for ext in FACT_EXTRACTORS:
                res = ext(candidate)
                if res is not None:
                    positive_facts.append(res)
                    
            # Sort positive facts by score descending
            positive_facts.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 2 positive facts text
            facts_text_list = [x[0] for x in positive_facts]
            if len(facts_text_list) >= 2:
                facts_text = f"{facts_text_list[0]}; {facts_text_list[1]}"
            elif len(facts_text_list) == 1:
                facts_text = f"{facts_text_list[0]}"
            else:
                facts_text = "profile reviewed"
                
            # Get concern
            concern = get_concern_fact(candidate)
            
            # Rank-based Tone Prefix (SPEC-7.6)
            if 1 <= rank <= 5:
                prefix = "Top fit:"
            elif 6 <= rank <= 20:
                prefix = "Strong match:"
            elif 21 <= rank <= 50:
                prefix = "Solid candidate:"
            elif 51 <= rank <= 80:
                prefix = "Marginal fit:"
            else:
                prefix = "Below cutoff but included:"
                
            reasoning = f"{prefix} {facts_text}. Concern: {concern}."
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])
            
    logger.info(f"=== Pipeline Complete. Top candidate: {top_candidates[0][0]} with score {top_candidates[0][1]:.6f} ===")

if __name__ == "__main__":
    main()