# src/utils/config.py
import os
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Load variables from .env file into environment
load_dotenv()

class Config:
    """Central configuration for file paths and constants."""
    
    # Detect environment, default to 'prod' for safety
    APP_ENV = os.getenv("APP_ENV", "prod").lower()
    
    if APP_ENV == "test":
        logger.info("Running in TEST environment. Using mock data.")
        # Paths for local testing
        RAW_DATA_PATH = "tests/fixtures/data/mock_candidates.jsonl"
        INTERIM_DATA_PATH = "tests/fixtures/interim/mock_candidates.jsonl"
        EMBEDDINGS_PATH = "tests/fixtures/interim/mock_embeddings.npy"
        CANDIDATE_IDS_PATH = "tests/fixtures/interim/mock_ids.npy"
        JD_PARAPHRASE_PATH = "tests/fixtures/data/mock_jd.txt"
        JD_EMBEDDING_PATH = "tests/fixtures/interim/mock_jd_embedding.npy"
    else:
        logger.info("Running in PROD environment. Using real data paths.")
        # Paths for production / hackathon submission
        RAW_DATA_PATH = "data/raw/candidates.jsonl.gz"
        INTERIM_DATA_PATH = "data/interim/candidates.jsonl"
        EMBEDDINGS_PATH = "data/interim/candidate_embeddings.npy"
        CANDIDATE_IDS_PATH = "data/interim/candidate_ids.npy"
        JD_PARAPHRASE_PATH = "data/raw/jd_paraphrased.txt"
        JD_EMBEDDING_PATH = "data/interim/jd_embedding.npy"
    
    # Model settings (constant across environments)
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 64