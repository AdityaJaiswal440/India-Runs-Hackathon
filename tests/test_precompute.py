# tests/test_precompute.py
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
from scripts import precompute

@pytest.fixture
def mock_candidates_data():
    return [
        {
            "candidate_id": "CAND_1",  # The Winner
            "profile": {
                "headline": "ML Engineer",
                "summary": "Built ranking systems.",
            },
            "career_history": [
                {"description": "Shipped retrieval systems at Swiggy."}
            ],
            "redrob_signals": {
                "recruiter_response_rate": 0.9,
                "notice_period_days": 30,
                "last_active_date": "2025-12-01",
            },
        },
        {
            "candidate_id": "CAND_2",  # The Loser
            "profile": {
                "headline": "Marketing Manager",
                "summary": "Ran Facebook ads.",
            },
            "career_history": [{"description": "Managed ad spend."}],
            "redrob_signals": {
                "recruiter_response_rate": 0.1,
                "notice_period_days": 120,
                "last_active_date": "2025-01-01",
            },
        },
        {
            "candidate_id": "CAND_3",  # <-- THE MATHEMATICAL BALLAST
            "profile": {"headline": "Chef", "summary": "Bakes cakes."},
            "career_history": [{"description": "Operated an oven."}],
            "redrob_signals": {
                "recruiter_response_rate": 0.5,
                "notice_period_days": 60,
                "last_active_date": "2025-06-01",
            },
        },
    ]

def test_simple_tokenize():
    """Test that tokenizer strips punctuation and lowercases"""
    text = "Hello, World! Built RAG & Pinecone."
    tokens = precompute.simple_tokenize(text)
    assert tokens == ["hello", "world", "built", "rag", "pinecone"]

@patch('scripts.precompute.Embedder')
@patch('scripts.precompute.is_honeypot', return_value=False)
@patch('scripts.precompute.stream_candidates')
@patch('scripts.precompute.Config')
def test_run_precompute_pipeline(mock_config, mock_stream, mock_honeypot, mock_embedder_class, tmp_path, mock_candidates_data):
    """Test the full precompute orchestration with mocked models"""
    
    # 1. Setup Mocks
    mock_stream.return_value = mock_candidates_data
    
    # Mock the embedder instance and its methods
    mock_embedder_instance = MagicMock()
    # Return a (1, 384) array for JD, (3, 384) array for candidates
    mock_embedder_instance.embed_texts.side_effect = [
        np.ones((1, 384)), # jd_vector
        np.ones((3, 384))  # candidate_vectors
    ]
    mock_embedder_class.return_value = mock_embedder_instance
    
    # Mock config paths to use pytest's tmp_path
    mock_config.JD_REQUIREMENTS_PATH = str(tmp_path / "jd.txt")
    mock_config.FEATURES_TABLE_PATH = str(tmp_path / "features.parquet")
    
    # Create dummy JD file for the script to read
    with open(mock_config.JD_REQUIREMENTS_PATH, 'w') as f:
        f.write("ranking systems retrieval systems")
        
    # 2. Run the pipeline
    precompute.run_precompute()
    
    # Check that stream_candidates was called
    assert mock_stream.call_count >= 1
    
    # Check that embedder was called twice (once for candidates, once for JD)
    assert mock_embedder_instance.embed_texts.call_count == 2
    
    # Check that the Parquet file was created
    assert os.path.exists(mock_config.FEATURES_TABLE_PATH)
    
    # Read the Parquet and verify contents
    df = pd.read_parquet(mock_config.FEATURES_TABLE_PATH)
    
    assert len(df) == 3
    assert "candidate_id" in df.columns
    assert "bm25_score" in df.columns
    assert "dense_sim_score" in df.columns
    assert "is_honeypot" in df.columns
    
    # Candidate 1 should have a higher BM25 score than Candidate 2
    # because JD has "ranking" and "retrieval", and C1 text has them too.
    assert df.loc[0, "bm25_score"] > df.loc[1, "bm25_score"]
    
    # Dense sim should be 384 (1 * 1 dot product)
    assert df.loc[0, "dense_sim_score"] == pytest.approx(384.0)