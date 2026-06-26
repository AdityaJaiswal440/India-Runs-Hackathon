# tests/test_heuristic_extractor.py
import pytest
import datetime
from src.features.heuristic_extractor import behavioral_multiplier

def test_behavioral_multiplier_perfect():
    candidate = {
        "profile": {
            "current_title": "Software Engineer",
        },
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": "2025-12-01",
            "recruiter_response_rate": 1.0,
            "avg_response_time_hours": 12.0,
            "notice_period_days": 10,
            "verified_email": True,
            "verified_phone": True,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 1.0,
            "github_activity_score": 50,
            "profile_completeness_score": 100,
            "linkedin_connected": True
        }
    }
    today = datetime.date(2025, 12, 10) # 9 days inactive -> no penalty
    m = behavioral_multiplier(candidate, today)
    assert abs(m - 1.0) < 1e-6

def test_behavioral_multiplier_penalties():
    candidate = {
        "profile": {
            "current_title": "Principal Architect",
        },
        "redrob_signals": {
            "open_to_work_flag": False,
            "last_active_date": "2024-01-01",
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 120,
            "notice_period_days": 120,
            "verified_email": False,
            "verified_phone": False,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": 0.8,
            "github_activity_score": 5,
            "profile_completeness_score": 40,
            "linkedin_connected": False
        }
    }
    today = datetime.date(2026, 1, 1) # 2 years inactive (731 days)
    m = behavioral_multiplier(candidate, today)
    assert m == 0.05