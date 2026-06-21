# tests/test_heuristic_extractor.py
import pytest
from src.features.heuristic_extractor import extract_features

@pytest.fixture
def mock_product_candidate():
    return {
        "candidate_id": "CAND_1",
        "profile": {"years_of_experience": 7},
        "career_history": [
            {"company": "Swiggy", "description": "Built ranking systems."},
            {"company": "Zomato", "description": "Built retrieval systems."}
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.9,
            "notice_period_days": 30,
            "last_active_date": "2025-12-01"
        }
    }

@pytest.fixture
def mock_consulting_candidate():
    return {
        "candidate_id": "CAND_2",
        "profile": {"years_of_experience": 8},
        "career_history": [
            {"company": "TCS", "description": "Did IT work."},
            {"company": "Infosys", "description": "Did IT work."}
        ],
        "redrob_signals": {
            "recruiter_response_rate": 0.1,
            "notice_period_days": 120,
            "last_active_date": "2025-01-01"
        }
    }

@pytest.fixture
def mock_empty_history_candidate():
    return {
        "candidate_id": "CAND_3",
        "profile": {"years_of_experience": 2},
        "career_history": [], # Empty history
        "redrob_signals": {
            "recruiter_response_rate": 0.5,
            "notice_period_days": 60,
            "last_active_date": "2025-11-01"
        }
    }

def test_product_candidate_features(mock_product_candidate):
    feats = extract_features(mock_product_candidate)
    assert feats["yoe"] == 7
    assert feats["yoe_distance"] == 0
    assert feats["is_consulting_only"] == False
    assert feats["behavior_score"] == 1.0 # Perfect behavior

def test_consulting_candidate_features(mock_consulting_candidate):
    feats = extract_features(mock_consulting_candidate)
    assert feats["is_consulting_only"] == True
    # 0.1 response (-0.2), 120 day notice (-0.2), stale active (-0.3) = 0.3
    assert feats["behavior_score"] == 0.5 # Clamped to minimum

def test_empty_history_candidate_features(mock_empty_history_candidate):
    """Ensure extractor doesn't crash on empty career history"""
    feats = extract_features(mock_empty_history_candidate)
    assert feats["is_consulting_only"] == False
    assert feats["yoe"] == 2
    assert feats["yoe_distance"] == 3 # 5 - 2 = 3