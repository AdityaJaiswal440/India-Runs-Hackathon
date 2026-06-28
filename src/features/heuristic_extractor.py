"""Heuristic feature configuration — skill taxonomies and JD constants.

.. deprecated::
    All constants in this module have been migrated to
    ``src/challenge/jd_config.py``. Import from there directly.
    This shim is retained only for backwards compatibility with
    any external tooling that references it; it will be removed
    in a future cleanup commit.
"""

from __future__ import annotations

from src.challenge.jd_config import (
    CONSULTING_FIRMS as CONSULTING_FIRMS,
    CORE_SKILL_PHRASES as _CORE_TUPLE,
    SECONDARY_SKILL_PHRASES as _SEC_TUPLE,
)

# Legacy set-based aliases — jd_config uses tuples for ordering stability
CORE_SKILL_PHRASES = frozenset(_CORE_TUPLE)
SECONDARY_SKILL_PHRASES = frozenset(_SEC_TUPLE)
TIER_1_SKILLS = CORE_SKILL_PHRASES | SECONDARY_SKILL_PHRASES

# Legacy flat weight dict used by audit_submission.py grounding checks
JD_TAXONOMY: dict[str, float] = {
    **{skill: 1.0 for skill in CORE_SKILL_PHRASES},
    **{skill: 0.5 for skill in SECONDARY_SKILL_PHRASES},
}

# Legacy stubs to prevent scripts/precompute.py and sandbox/app.py from crashing
def compute_skill_match_score(candidate): return 0.0
def compute_skill_gap_coverage(candidate): return 0.0
def compute_career_score(candidate): return 0.0
def compute_experience_fit(candidate): return 0.0
def compute_location_score(candidate): return 0.0
def compute_education_score(candidate): return 0.0
def compute_engagement_score(candidate): return 0.0
def behavioral_multiplier(candidate): return 1.0
def extract_features(candidate): return {"yoe_distance": 0.0, "is_consulting_only": False, "adjusted_fit": 0.0}