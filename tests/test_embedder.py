# tests/test_embedder.py
import numpy as np
import pytest
from src.features.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    """
    Loads the HuggingFace model once for the whole test file, 
    rather than reloading it for every individual test.
    """
    return Embedder()

def test_embedder_output_dimensions(embedder):
    """Asserts that two input sentences produce a (2, 384) matrix."""
    sample_texts = [
        "Senior Data Scientist experienced in vector search.",
        "Python backend engineer with FastAPI skills."
    ]
    
    result = embedder.embed_texts(sample_texts)
    
    assert isinstance(result, np.ndarray), "Output must be a numpy array"
    assert result.shape == (2, 384), f"Expected shape (2, 384), got {result.shape}"
    assert not np.isnan(result).any(), "Embeddings contain NaN values"

def test_embedder_empty_input_handling(embedder):
    """Asserts that passing empty lists doesn't crash the HuggingFace C-bindings."""
    result = embedder.embed_texts([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0