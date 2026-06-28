"""Candidate availability scoring.

The JD explicitly down-weights candidates who are not actively job-seeking
or who are unresponsive to recruiter outreach. This module converts raw
redrob_signals into a composite availability score and a strong ranking
modifier — availability can swing a candidate's final position significantly.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict

from config.defaults import RANKING_REFERENCE_DATE

_REFERENCE_DATE: date = RANKING_REFERENCE_DATE


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def availability_score(signals: Dict[str, Any]) -> float:
    """Composite availability in [0, 1].

    Component weights (must sum to 1.0):
      open_to_work      25%  — binary, strongest single gate
      recency           20%  — days since last_active_date
      applications      10%  — applications submitted in last 30 days
      recruiter_rr      20%  — recruiter response rate
      interview_cr      10%  — interview completion rate
      response_time     10%  — avg response time in hours (inverted)
      offer_acceptance   5%  — offer acceptance rate
    """
    open_to_work = bool(signals.get("open_to_work_flag", False))
    last_active = _parse_date(signals.get("last_active_date"))
    apps = int(signals.get("applications_submitted_30d", 0) or 0)
    rr = float(signals.get("recruiter_response_rate", 0) or 0)
    icr = float(signals.get("interview_completion_rate", 0) or 0)
    rth = float(signals.get("avg_response_time_hours", 0) or 0)
    oar = float(signals.get("offer_acceptance_rate", -1) or -1)

    days_idle = (_REFERENCE_DATE - last_active).days if last_active else 365

    recency = max(0.0, 1.0 - days_idle / 180.0)
    otw_s = 1.0 if open_to_work else 0.04
    apps_s = min(1.0, apps / 4.0)
    rr_s = min(1.0, rr * 1.15)
    icr_s = min(1.0, icr)
    rth_s = min(1.0, 36.0 / rth) if rth > 0 else 0.35
    oar_s = min(1.0, max(0.0, oar)) if oar >= 0 else 0.5

    return min(
        1.0,
        0.25 * otw_s
        + 0.20 * recency
        + 0.10 * apps_s
        + 0.20 * rr_s
        + 0.10 * icr_s
        + 0.10 * rth_s
        + 0.05 * oar_s,
    )


def availability_modifier(signals: Dict[str, Any]) -> float:
    """Multiplicative modifier applied to the candidate's final score.

    Hard gate: open_to_work=False collapses the modifier to 0.01, effectively
    removing the candidate from contention regardless of other signals.
    """
    open_to_work = bool(signals.get("open_to_work_flag", False))
    if not open_to_work:
        return 0.01

    s = availability_score(signals)
    if s < 0.2:
        return 0.42
    if s < 0.35:
        return 0.58
    if s < 0.5:
        return 0.75
    if s < 0.65:
        return 0.88
    return 0.90 + s * 0.10
