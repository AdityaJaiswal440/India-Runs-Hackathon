"""Honeypot detection — structural traps, not title-keyword matching."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

from config.defaults import HONEYPOT_REFERENCE_DATE
from challenge.text_match import norm_text

_REFERENCE_DATE = HONEYPOT_REFERENCE_DATE


def _parse_date(value: str | None) -> date | None:
    if not value or len(value) < 10:
        return None
    try:
        return date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        return None


def _career_span_years(history: List[Dict[str, Any]]) -> float:
    if not history:
        return 0.0
    starts: list[date] = []
    ends: list[date] = []
    for role in history:
        start = _parse_date(role.get("start_date"))
        if not start:
            continue
        starts.append(start)
        end = _parse_date(role.get("end_date")) if role.get("end_date") else _REFERENCE_DATE
        ends.append(end or _REFERENCE_DATE)
    if not starts:
        return 0.0
    earliest = min(starts)
    latest = max(ends)
    return max(0.0, (latest - earliest).days / 365.25)


def honeypot_risk(raw: Dict[str, Any]) -> float:
    """
    Returns 0.0 (clean) → 1.0 (certain trap).

    Rules:
    1. Expert skills with zero months used (documented signature)
    2. Claimed YoE >> calendar span of career_history
    3. Overlapping roles exceeding plausible concurrent employment
    4. Expert skill stuffing with thin duration/endorsement mismatch
    5. Suspicious endorsement density on unverified skills
    """
    profile = raw.get("profile", {})
    skills: List[Dict[str, Any]] = raw.get("skills", [])
    history: List[Dict[str, Any]] = raw.get("career_history", [])
    risk = 0.0

    # Rule 1 — expert @ 0 months
    expert_zero = [
        s
        for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= 6:
        risk = max(risk, 0.95)
    elif len(expert_zero) >= 4:
        risk = max(risk, 0.80)
    elif len(expert_zero) >= 2:
        risk = max(risk, 0.50)

    # Rule 2 — impossible tenure vs career span
    claimed = float(profile.get("years_of_experience", 0) or 0)
    span = _career_span_years(history)
    if span > 0 and claimed > span * 1.25 + 1.5:
        gap = claimed - span
        risk = max(risk, min(0.98, 0.55 + gap * 0.06))

    # Rule 3 — overlapping roles (sum durations >> span)
    if history:
        total_months = sum(int(h.get("duration_months", 0) or 0) for h in history)
        span_months = max(1.0, span * 12)
        if total_months > span_months * 1.6 and span > 2:
            risk = max(risk, 0.6)

    # Rule 4 — skill stuffing: many expert/advanced, low average duration
    expert_adv = [
        s for s in skills if (s.get("proficiency") or "").lower() in ("expert", "advanced")
    ]
    if len(expert_adv) >= 10:
        avg_dur = sum(int(s.get("duration_months", 0) or 0) for s in expert_adv) / len(expert_adv)
        if avg_dur < 8:
            risk = max(risk, 0.75)

    # Rule 5 — high endorsements + zero duration on same skill
    for s in skills:
        endorse = int(s.get("endorsements", 0) or 0)
        months = int(s.get("duration_months", 0) or 0)
        prof = (s.get("proficiency") or "").lower()
        if endorse >= 25 and months == 0 and prof in ("expert", "advanced"):
            risk = max(risk, 0.65)

    # Rule 6 — date contradictions (end before start)
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start and end and end < start:
            risk = max(risk, 0.8)

    # Rule 6b — Individual job duration vs calendar time check (assimilated from Protienbar)
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) if role.get("end_date") else _REFERENCE_DATE
        duration = int(role.get("duration_months", 0) or 0)
        if start and end and duration > 0:
            calendar_months = (end.year - start.year) * 12 + (end.month - start.month)
            if duration > calendar_months + 2:
                risk = max(risk, 0.85)

    # Rule 7 — education end year vs claimed experience inconsistency
    education = raw.get("education") or []
    if education and claimed > 0:
        end_years = [int(e.get("end_year", 0) or 0) for e in education if e.get("end_year")]
        if end_years:
            latest_grad = max(end_years)
            min_possible = _REFERENCE_DATE.year - latest_grad
            if claimed > min_possible + 8:
                risk = max(risk, min(0.85, 0.45 + (claimed - min_possible) * 0.04))

    # Rule 8 — tenure at current company exceeds total career span
    if history:
        current = next((h for h in history if h.get("is_current")), history[0])
        cur_months = int(current.get("duration_months", 0) or 0)
        if span > 0 and cur_months > span * 12 * 1.15:
            risk = max(risk, 0.7)

    # Rule 9 — implausible junior title + mega-corp + stuffed expert skills
    title = norm_text(profile.get("current_title", ""))
    company_size = norm_text(profile.get("current_company_size", ""))
    if ("junior" in title or "intern" in title) and "10001" in company_size:
        expert_n = sum(
            1 for s in skills if (s.get("proficiency") or "").lower() == "expert"
        )
        if expert_n >= 8:
            risk = max(risk, 0.75)

    # Rule 10 — YoE vs Actual Career Duration discrepancy (assimilated from Praveen-ing)
    total_months = sum(int(job.get("duration_months", 0) or 0) for job in history)
    career_yoe = total_months / 12.0
    if claimed > 5.0 and career_yoe < 1.0:
        risk = max(risk, 0.9)
    if career_yoe > claimed + 5.0:
        risk = max(risk, 0.9)

    # Rule 11 — Shannon Entropy Anomaly Checker (assimilated from Praveen-ing)
    candidate_words = []
    for s in skills:
        if s and s.get("name"):
            candidate_words.append(s["name"].lower())
    for job in history:
        if job:
            candidate_words.append((job.get("title") or "").lower())
            candidate_words.append((job.get("description") or "").lower())
    full_profile_text = " ".join(candidate_words)
    
    import math
    import re
    words = re.findall(r'\b\w+\b', full_profile_text)
    if words:
        total_words = len(words)
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        entropy = 0.0
        for count in counts.values():
            p = count / total_words
            entropy -= p * math.log2(p)
            
        if entropy < 2.2:
            risk = max(risk, 0.95)

    return min(1.0, risk)


def risk_to_penalty(risk: float) -> float:
    """Map honeypot_risk → multiplicative demotion (compute once per candidate)."""
    if risk >= 0.9:
        return 0.02
    if risk >= 0.75:
        return 0.08
    if risk >= 0.55:
        return 0.22
    if risk >= 0.35:
        return 0.45
    if risk >= 0.2:
        return 0.72
    return max(0.85, 1.0 - risk * 0.35)


def honeypot_penalty(raw: Dict[str, Any]) -> float:
    return risk_to_penalty(honeypot_risk(raw))


def is_structural_honeypot(raw: Dict[str, Any], threshold: float = 0.55) -> bool:
    return honeypot_risk(raw) >= threshold


def is_keyword_stuffer(raw: Dict[str, Any]) -> bool:
    profile = raw.get("profile", {})
    skills = raw.get("skills", []) or []
    history = raw.get("career_history", []) or []
    signals = raw.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}

    current_title = (profile.get("current_title") or "").lower().strip()
    
    # 1. Define title classes
    RELEVANT_TITLE_TERMS = {
        "ml engineer", "machine learning", "ai engineer", "applied scientist",
        "applied ml", "nlp engineer", "data scientist", "search engineer",
        "recommendation systems", "recsys", "research engineer", "ai research",
        "ai specialist", "deep learning"
    }
    ADJACENT_TITLE_TERMS = {
        "software engineer", "backend engineer", "data engineer", "analytics engineer",
        "full stack", "platform engineer", "staff engineer", "senior software"
    }
    NONTECH_TITLE_TERMS = {
        "hr manager", "marketing manager", "content writer", "graphic designer",
        "accountant", "sales executive", "customer support", "operations manager",
        "business analyst", "project manager", "civil engineer", "mechanical engineer",
        "qa engineer"
    }
    OFFDOMAIN_TITLE_TERMS = {"computer vision", "cv engineer", "speech", "robotics"}
    
    def get_title_class(title):
        t = title.lower()
        if any(term in t for term in RELEVANT_TITLE_TERMS):
            return "relevant"
        if any(term in t for term in OFFDOMAIN_TITLE_TERMS):
            return "offdomain"
        if any(term in t for term in NONTECH_TITLE_TERMS):
            return "nontech"
        if any(term in t for term in ADJACENT_TITLE_TERMS):
            return "adjacent"
        return "other"

    current_class = get_title_class(current_title)
    if current_class not in ("nontech", "offdomain"):
        return False
        
    # 2. Count AI skills listed
    from src.features.heuristic_extractor import TIER_1_SKILLS
    ai_skill_count = sum(1 for s in skills if (s.get("name") or "").lower().strip() in TIER_1_SKILLS)
    if ai_skill_count < 3:
        return False

    # 3. Check if they ever held a tech role in their history or current title
    all_titles = [current_title] + [(job.get("title") or "").lower().strip() for job in history]
    held_tech_role = any(
        any(term in t for term in RELEVANT_TITLE_TERMS) or any(term in t for term in ADJACENT_TITLE_TERMS)
        for t in all_titles
    )
    if held_tech_role:
        return False

    # 4. Count AI evidence in narrative (headline, summary, history descriptions)
    narrative = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(job.get("description", "") for job in history)
    ]).lower()
    
    evidence_count = sum(1 for term in TIER_1_SKILLS if term in narrative)
    if evidence_count >= 3:
        return False

    # 5. Check if they have any verified assessment >= 60
    has_verified = any(
        (val is not None and float(val) >= 60.0)
        for name, val in assessments.items()
        if name.lower().strip() in TIER_1_SKILLS
    )
    if has_verified:
        return False

    return True


def stuffer_penalty(raw: Dict[str, Any]) -> float:
    if is_keyword_stuffer(raw):
        return 0.05
    return 1.0