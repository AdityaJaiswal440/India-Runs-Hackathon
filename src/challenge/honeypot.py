"""Honeypot detection — structural consistency checks for candidate profiles.

Scores each candidate on a continuous risk scale: 0.0 (clean) → 1.0 (certain trap).
Rules operate on structural signals only (timeline math, endorsement density,
title/skill coherence) — not on keyword presence or domain matching.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Dict, List

from config.defaults import RANKING_REFERENCE_DATE
from challenge.text_match import norm_text
from challenge.company_matrix import COMPANY_FOUNDING_YEARS
from challenge.jd_config import CORE_SKILL_PHRASES, SECONDARY_SKILL_PHRASES

_REFERENCE_DATE: date = RANKING_REFERENCE_DATE

# Derived skill set used by the keyword-stuffer detector
_TIER_1_SKILLS: frozenset[str] = frozenset(CORE_SKILL_PHRASES) | frozenset(SECONDARY_SKILL_PHRASES)

_SUFFIX_PATTERN = re.compile(
    r"\b(inc|ltd|pvt|llc|corp|technologies|tech|software|solutions|private|limited)\b",
    re.IGNORECASE,
)

# Known placeholder company names produced by standard synthetic data generators.
# Implemented as an O(1) set for fast short-circuit evaluation before heavier rules.
_SYNTHETIC_COMPANY_BLOCKLIST: frozenset[str] = frozenset({
    "piedpiper",
    "initech",
    "wayneenterprises",
    "acme",
    "starkindustries",
    "hooli",
    "globex",
    "dundermifflin",
})

# Minimum release year (fractional, month/12) for tools where duration claims
# can be validated against public release history.
_TECH_RELEASE_YEAR: dict[str, float] = {
    "rag": 2020.4,
    "retrieval augmented generation": 2020.4,
    "langchain": 2022.8,
    "llamaindex": 2022.9,
    "qlora": 2023.4,
    "lora": 2021.5,
    "peft": 2023.1,
    "chatgpt": 2022.9,
    "gpt-4": 2023.2,
    "llama": 2023.1,
    "mistral": 2023.7,
    "pinecone": 2021.0,
    "qdrant": 2021.4,
    "prompt engineering": 2022.0,
    "stable diffusion": 2022.6,
}

# Phrases in career descriptions that indicate delegation rather than ownership.
_DELEGATION_PHRASES: tuple[str, ...] = (
    "deployment was handled by the platform team",
    "deployment was handled by",
    "production deployment was handled",
    "my role was more on modeling",
    "my role was more focused on modeling",
    "still building depth on the engineering side",
    "still developing depth on the engineering side",
    "primarily on the research side",
    "i was not responsible for",
    "handled by the infra team",
    "handled by the infrastructure team",
    "handled by ops",
    "engineering was owned by",
)


def _clean_company_name(name: str) -> str:
    """Strip legal suffixes and punctuation for consistent dict lookup."""
    if not name:
        return ""
    n = name.lower()
    n = _SUFFIX_PATTERN.sub("", n)
    n = re.sub(r"[^\w]", "", n)
    return n.strip()


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO-8601 date string, returning None on any failure."""
    if not value or len(value) < 10:
        return None
    try:
        return date(int(value[0:4]), int(value[5:7]), int(value[8:10]))
    except ValueError:
        return None


def safe_date(date_str: Any) -> date | None:
    """Public helper: parse a date value that may be a string, date, or datetime."""
    if not date_str:
        return None
    try:
        if isinstance(date_str, date):
            return date_str
        import datetime as _dt
        if isinstance(date_str, _dt.datetime):
            return date_str.date()
        return date.fromisoformat(str(date_str).strip())
    except (ValueError, TypeError):
        return None


def _career_span_years(history: List[Dict[str, Any]]) -> float:
    """Compute calendar span (years) from earliest start to latest end across all roles."""
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
    return max(0.0, (max(ends) - min(starts)).days / 365.25)


def _overlap_months(entry1: Dict[str, Any], entry2: Dict[str, Any]) -> int:
    """Return the number of months two job entries overlap, or 0 if they do not."""
    s1 = _parse_date(entry1.get("start_date"))
    e1 = _parse_date(entry1.get("end_date")) or _parse_date(entry1.get("start_date"))
    s2 = _parse_date(entry2.get("start_date"))
    e2 = _parse_date(entry2.get("end_date")) or _parse_date(entry2.get("start_date"))
    if not (s1 and e1 and s2 and e2):
        return 0
    start1, end1 = s1.year * 12 + s1.month, e1.year * 12 + e1.month
    start2, end2 = s2.year * 12 + s2.month, e2.year * 12 + e2.month
    return max(0, min(end1, end2) - max(start1, start2))


def honeypot_risk(raw: Dict[str, Any]) -> float:
    """Return a risk score in [0.0, 1.0] — higher means more likely synthetic/fraudulent.

    Rules are applied in order of computational cost (cheapest first).
    Any rule can raise the risk ceiling; the final value is min(1.0, risk).
    """
    profile = raw.get("profile", {})
    skills: List[Dict[str, Any]] = raw.get("skills", []) or []
    history: List[Dict[str, Any]] = raw.get("career_history", []) or []
    signals: Dict[str, Any] = raw.get("redrob_signals", {}) or {}
    education = raw.get("education") or []

    # Rule 27 — Synthetic placeholder company (O(1) short-circuit)
    for job in history:
        company_raw = job.get("company")
        if company_raw and _clean_company_name(company_raw) in _SYNTHETIC_COMPANY_BLOCKLIST:
            return 1.0

    risk = 0.0

    claimed = float(profile.get("years_of_experience", 0) or 0)
    span = _career_span_years(history)

    # Rule 1 — Expert-level skills with zero duration months claimed
    expert_zero = [
        s for s in skills
        if (s.get("proficiency") or "").lower() == "expert"
        and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= 6:
        risk = max(risk, 0.95)
    elif len(expert_zero) >= 4:
        risk = max(risk, 0.80)
    elif len(expert_zero) >= 2:
        risk = max(risk, 0.50)

    # Rule 2 — Claimed YoE significantly exceeds calendar span of career history
    if span > 0 and claimed > span * 1.25 + 1.5:
        gap = claimed - span
        risk = max(risk, min(0.98, 0.55 + gap * 0.06))

    # Rule 3 — Sum of role durations far exceeds career calendar span (overlapping roles)
    if history:
        total_months = sum(int(h.get("duration_months", 0) or 0) for h in history)
        span_months = max(1.0, span * 12)
        if total_months > span_months * 1.6 and span > 2:
            risk = max(risk, 0.6)

    # Rule 4 — Many expert/advanced skills with suspiciously low average duration
    expert_adv = [
        s for s in skills
        if (s.get("proficiency") or "").lower() in ("expert", "advanced")
    ]
    if len(expert_adv) >= 10:
        avg_dur = sum(int(s.get("duration_months", 0) or 0) for s in expert_adv) / len(expert_adv)
        if avg_dur < 8:
            risk = max(risk, 0.75)

    # Rule 5 — High endorsements with zero duration on an expert-level skill
    for s in skills:
        endorse = int(s.get("endorsements", 0) or 0)
        months = int(s.get("duration_months", 0) or 0)
        prof = (s.get("proficiency") or "").lower()
        if endorse >= 25 and months == 0 and prof in ("expert", "advanced"):
            risk = max(risk, 0.65)

    # Rule 6 — End date precedes start date within a single role
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start and end and end < start:
            risk = max(risk, 0.8)

    # Rule 6b — Individual role duration_months exceeds its own calendar window
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) if role.get("end_date") else _REFERENCE_DATE
        duration = int(role.get("duration_months", 0) or 0)
        if start and end and duration > 0:
            calendar_months = (end.year - start.year) * 12 + (end.month - start.month)
            if duration > calendar_months + 2:
                risk = max(risk, 0.85)

    # Rule 7 — Claimed YoE impossible given latest education graduation year
    if education and claimed > 0:
        end_years = [int(e.get("end_year", 0) or 0) for e in education if e.get("end_year")]
        if end_years:
            latest_grad = max(end_years)
            min_possible = _REFERENCE_DATE.year - latest_grad
            if claimed > min_possible + 8:
                risk = max(risk, min(0.85, 0.45 + (claimed - min_possible) * 0.04))

    # Rule 8 — Tenure at current company exceeds total career span
    if history:
        current = next((h for h in history if h.get("is_current")), history[0])
        cur_months = int(current.get("duration_months", 0) or 0)
        if span > 0 and cur_months > span * 12 * 1.15:
            risk = max(risk, 0.7)

    # Rule 9 — Junior title at a mega-corp with an implausibly large expert skill count
    title = norm_text(profile.get("current_title", ""))
    company_size = norm_text(profile.get("current_company_size", ""))
    if ("junior" in title or "intern" in title) and "10001" in company_size:
        expert_n = sum(1 for s in skills if (s.get("proficiency") or "").lower() == "expert")
        if expert_n >= 8:
            risk = max(risk, 0.75)

    # Rule 10 — Claimed YoE vs. total months across all roles are mutually inconsistent
    total_months = sum(int(job.get("duration_months", 0) or 0) for job in history)
    career_yoe = total_months / 12.0
    if claimed > 5.0 and career_yoe < 1.0:
        risk = max(risk, 0.9)
    if career_yoe > claimed + 5.0:
        risk = max(risk, 0.9)

    # Rule 11 — Shannon entropy of profile text below threshold (repetitive/template content)
    candidate_words: list[str] = []
    for s in skills:
        if s and s.get("name"):
            candidate_words.append(s["name"].lower())
    for job in history:
        if job:
            candidate_words.append((job.get("title") or "").lower())
            candidate_words.append((job.get("description") or "").lower())
    full_text = " ".join(candidate_words)
    words = re.findall(r"\b\w+\b", full_text)
    if words:
        total_words = len(words)
        counts: dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        entropy = -sum((c / total_words) * math.log2(c / total_words) for c in counts.values())
        if entropy < 2.2:
            risk = max(risk, 0.95)

    # Rule 12 — Claimed YoE exceeds any plausible career length
    if claimed > 40:
        risk = max(risk, 0.95)

    # Rule 13 — All engagement signals simultaneously at maximum (statistically impossible)
    completeness = float(signals.get("profile_completeness_score", 0) or 0)
    linkedin = bool(signals.get("linkedin_connected", False))
    rr = float(signals.get("recruiter_response_rate", 0) or 0)
    saves = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    verified_email = bool(signals.get("email_verified", False))
    verified_phone = bool(signals.get("phone_verified", False))
    if completeness == 100 and verified_email and verified_phone and linkedin and saves > 50 and rr == 1.0:
        risk = 1.0

    # Rule 14 — Two or more roles overlap by 6+ months (impossible concurrent employment)
    for i in range(len(history)):
        for j in range(i + 1, len(history)):
            if _overlap_months(history[i], history[j]) >= 6:
                risk = max(risk, 0.75)

    # Rule 15 — Senior title with long history but almost no role descriptions
    senior_keywords = ("senior", "lead", "principal", "manager", "director")
    high_title = any(k in title for k in senior_keywords)
    long_history = (claimed > 5.0) if claimed > 0 else False
    if not long_history and history:
        total_months_hist = sum(int(job.get("duration_months", 0) or 0) for job in history)
        long_history = (total_months_hist / 12.0) >= 5.0
    descriptions = sum(1 for e in history if (e.get("description") or "").strip())
    if high_title and long_history and descriptions < max(1, len(history) // 3):
        risk = max(risk, 0.8)

    # Rule 16 — Title seniority level inconsistent with claimed years of experience
    curr_title = profile.get("current_title", "").lower().strip()
    if claimed < 3 and "senior" in curr_title:
        risk = max(risk, 0.85)
    if claimed < 5 and ("staff" in curr_title or "lead" in curr_title):
        risk = max(risk, 0.85)
    if claimed < 7 and "principal" in curr_title:
        risk = max(risk, 0.85)

    # Rule 17 — High recruiter saves but near-zero response or interview completion rate
    interview_rate = float(
        signals.get("interview_completion_rate", 1.0)
        if signals.get("interview_completion_rate") is not None
        else 1.0
    )
    if saves > 20 and (rr < 0.05 or interview_rate < 0.10):
        risk = max(risk, 0.90)

    # Rule 18 — Very junior candidate with implausibly high GitHub activity score
    gh_score = float(signals.get("github_activity_score", 0) or 0)
    if claimed < 1.0 and gh_score > 95:
        risk = max(risk, 0.85)

    # Rule 19 — A skill's duration exceeds the candidate's total claimed experience
    yoe_months = claimed * 12
    if skills and claimed > 0:
        max_skill_months = max(
            (int(s.get("duration_months", 0) or 0) for s in skills), default=0
        )
        if max_skill_months > yoe_months * 1.4:
            risk = max(risk, 0.10)

    # Rule 20 — Impossibly high endorsement count for someone with very little experience
    total_endorsements = sum(int(s.get("endorsements", 0) or 0) for s in skills)
    if total_endorsements > 5000 and claimed < 3:
        risk = max(risk, 0.10)

    # Rule 21 — Temporal anomalies: future start dates, current+end_date contradiction, time travel
    temporal_violation = False
    for job in history:
        sd = _parse_date(job.get("start_date"))
        ed = _parse_date(job.get("end_date"))
        if sd and sd > _REFERENCE_DATE:
            temporal_violation = True
        if job.get("is_current") and job.get("end_date"):
            temporal_violation = True
        if sd and ed and ed < sd:
            temporal_violation = True
    if temporal_violation:
        risk = max(risk, 0.90)

    # Rule 22 — Assessment scores present for skills not listed in the skills section
    assessments: Dict[str, Any] = signals.get("skill_assessment_scores") or {}
    if assessments and skills:
        skill_names = {s.get("name", "").lower() for s in skills if s.get("name")}
        mismatches = sum(
            1 for assessment_skill in assessments
            if not any(
                assessment_skill.lower() in sn or sn in assessment_skill.lower()
                for sn in skill_names
            )
        )
        if mismatches >= 3:
            risk = max(risk, 0.85)

    # Rule 23 — Education end year precedes start year (reverse timeline)
    for edu in education:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        try:
            if sy and ey and int(ey) < int(sy):
                risk = max(risk, 0.90)
        except (ValueError, TypeError):
            pass

    # Rule 24 — Skill duration claim exceeds the technology's public existence
    ref_frac = _REFERENCE_DATE.year + _REFERENCE_DATE.month / 12.0
    anachronistic = 0
    for s in skills:
        sname = norm_text(s.get("name", ""))
        rel = _TECH_RELEASE_YEAR.get(sname)
        if rel:
            max_months = (ref_frac - rel) * 12 + 4  # 4-month grace period
            if int(s.get("duration_months", 0) or 0) > max_months:
                anachronistic += 1
    if anachronistic >= 2:
        risk = max(risk, 0.90)

    # Rule 25 — Career descriptions explicitly disclaim production ownership
    narrative = " ".join((job.get("description") or "") for job in history).lower()
    if any(phrase in narrative for phrase in _DELEGATION_PHRASES):
        risk = max(risk, 0.75)

    # Rule 26 — Candidate claims employment at a company before it was founded
    for job in history:
        company = _clean_company_name(job.get("company") or "")
        start_s = job.get("start_date")
        if start_s and company:
            try:
                start_year = int(start_s.split("-")[0])
                founding_year = COMPANY_FOUNDING_YEARS.get(company)
                if founding_year is not None and start_year < founding_year:
                    return 1.0
            except Exception:
                pass

    return min(1.0, risk)


def risk_to_penalty(risk: float) -> float:
    """Map a honeypot risk score to a multiplicative score demotion factor."""
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
    """Return the multiplicative score penalty for a candidate profile."""
    return risk_to_penalty(honeypot_risk(raw))


def is_structural_honeypot(raw: Dict[str, Any], threshold: float = 0.55) -> bool:
    """Return True if the profile's risk score meets or exceeds the given threshold."""
    return honeypot_risk(raw) >= threshold


def is_honeypot(
    raw: Dict[str, Any],
    today: date | None = None,
    clone_ids: set | None = None,
) -> bool:
    """Public API: evaluate a profile against honeypot risk at the default threshold."""
    cid = raw.get("candidate_id")
    if cid and clone_ids and cid in clone_ids:
        return True
    return honeypot_risk(raw) >= 0.55


def is_keyword_stuffer(raw: Dict[str, Any]) -> bool:
    """Detect non-technical candidates who have bulk-added AI skills to their profile.

    A stuffer is identified when all of the following hold:
      1. Current (and past) titles are in a non-technical or off-domain category.
      2. Three or more Tier-1 AI skills appear in the skills list.
      3. No tech role appears anywhere in the career history.
      4. Career narrative contains fewer than 3 AI skill mentions.
      5. No verified skill assessment score >= 60 for any IR/ML skill.
    """
    profile = raw.get("profile", {})
    skills = raw.get("skills", []) or []
    history = raw.get("career_history", []) or []
    signals = raw.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}

    current_title = (profile.get("current_title") or "").lower().strip()

    RELEVANT_TITLE_TERMS = {
        "ml engineer", "machine learning", "ai engineer", "applied scientist",
        "applied ml", "nlp engineer", "data scientist", "search engineer",
        "recommendation systems", "recsys", "research engineer", "ai research",
        "ai specialist", "deep learning",
    }
    ADJACENT_TITLE_TERMS = {
        "software engineer", "backend engineer", "data engineer", "analytics engineer",
        "full stack", "platform engineer", "staff engineer", "senior software",
    }
    NONTECH_TITLE_TERMS = {
        "hr manager", "marketing manager", "content writer", "graphic designer",
        "accountant", "sales executive", "customer support", "operations manager",
        "business analyst", "project manager", "civil engineer", "mechanical engineer",
        "qa engineer",
    }
    OFFDOMAIN_TITLE_TERMS = {"computer vision", "cv engineer", "speech", "robotics"}

    def _title_class(t: str) -> str:
        if any(term in t for term in RELEVANT_TITLE_TERMS):
            return "relevant"
        if any(term in t for term in OFFDOMAIN_TITLE_TERMS):
            return "offdomain"
        if any(term in t for term in NONTECH_TITLE_TERMS):
            return "nontech"
        if any(term in t for term in ADJACENT_TITLE_TERMS):
            return "adjacent"
        return "other"

    if _title_class(current_title) not in ("nontech", "offdomain"):
        return False

    ai_skill_count = sum(
        1 for s in skills if (s.get("name") or "").lower().strip() in _TIER_1_SKILLS
    )
    if ai_skill_count < 3:
        return False

    all_titles = [current_title] + [(job.get("title") or "").lower().strip() for job in history]
    held_tech_role = any(
        any(term in t for term in RELEVANT_TITLE_TERMS)
        or any(term in t for term in ADJACENT_TITLE_TERMS)
        for t in all_titles
    )
    if held_tech_role:
        return False

    narrative = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(job.get("description", "") for job in history),
    ]).lower()
    if sum(1 for term in _TIER_1_SKILLS if term in narrative) >= 3:
        return False

    has_verified = any(
        val is not None and float(val) >= 60.0
        for name, val in assessments.items()
        if name.lower().strip() in _TIER_1_SKILLS
    )
    if has_verified:
        return False

    return True


def stuffer_penalty(raw: Dict[str, Any]) -> float:
    """Return 0.05 multiplier for confirmed keyword stuffers, 1.0 otherwise."""
    return 0.05 if is_keyword_stuffer(raw) else 1.0