# src/features/heuristic_extractor.py
"""
Module: Heuristic Feature Extractor
Responsibility: Parse raw candidate JSON and extract structured, numerical, 
and boolean features (YoE, consulting flags, behavioral modifiers). 
Does NOT compute semantic or BM25 scores.
"""

from src.utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

# List of consulting firms to flag (JD disqualifier)
CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "hcl"]

def extract_features(candidate: dict) -> dict:
    """
    Extracts structured numerical and boolean features from a raw candidate dict.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    
    # 1. Experience Features
    yoe = profile.get("years_of_experience", 0)
    # Distance from the 5-9 sweet spot (0 if inside, positive if outside)
    if 5 <= yoe <= 9:
        yoe_distance = 0
    elif yoe < 5:
        yoe_distance = 5 - yoe
    else:
        yoe_distance = yoe - 9

    # 2. Consulting Flag (Hard Disqualifier proxy)
    is_consulting_only = False
    if history:
        # Check if ALL companies in history are consulting firms
        is_consulting_only = all(
            any(cfirm in job.get("company", "").lower() for cfirm in CONSULTING_FIRMS)
            for job in history
        )

    # 3. Behavioral Modifier (0.5 to 1.0 multiplier)
    # Start at 1.0, penalize for bad availability
    behavior_score = 1.0
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate < 0.3:
        behavior_score -= 0.2
        
    notice_period = signals.get("notice_period_days", 60)
    if notice_period > 90:
        behavior_score -= 0.2
        
    # Check last active date
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
            # Assuming competition date is roughly 2026-01-01 for calculation
            days_inactive = (datetime(2026, 1, 1) - last_active_dt).days
            if days_inactive > 180:
                behavior_score -= 0.3
        except ValueError:
            pass # Ignore malformed dates, honeypot detector will catch them

    # Clamp behavior score to minimum 0.5
    behavior_score = max(0.5, behavior_score)

    return {
        "yoe": yoe,
        "yoe_distance": yoe_distance,
        "is_consulting_only": is_consulting_only,
        "behavior_score": behavior_score,
        "response_rate": response_rate,
        "notice_period": notice_period
    }