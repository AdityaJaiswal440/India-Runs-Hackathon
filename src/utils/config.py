# src/utils/config.py
import os
import datetime
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

def load_dataset_today() -> datetime.date:
    """Loads the TODAY anchor date from artifacts/dataset_today.txt or fails loudly."""
    path = "artifacts/dataset_today.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required dataset today anchor file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        date_str = f.read().strip()
    return datetime.date.fromisoformat(date_str)

class Config:
    APP_ENV = os.getenv("APP_ENV", "prod").lower()
    
    if APP_ENV == "test":
        logger.info("Running in TEST environment. Using mock data.")
        RAW_DATA_PATH = "tests/fixtures/data/mock_candidates.jsonl"
        INTERIM_DATA_PATH = "tests/fixtures/interim/mock_candidates.jsonl"
        FEATURES_TABLE_PATH = "tests/fixtures/interim/mock_features.parquet"
        JD_FULL_PATH = "tests/fixtures/data/mock_jd_full.txt"
        JD_REQUIREMENTS_PATH = "tests/fixtures/data/mock_jd_requirements.txt"
        TEST_OUTPUT = "data/output/mock"
    else:
        logger.info("Running in PROD environment. Using real data paths.")
        RAW_DATA_PATH = "data/raw/candidates.jsonl.gz"
        INTERIM_DATA_PATH = "data/interim/candidates.jsonl"
        FEATURES_TABLE_PATH = "data/interim/features.parquet"
        JD_FULL_PATH = "data/raw/jd_paraphrased_full.txt"
        JD_REQUIREMENTS_PATH = "data/raw/jd_paraphrased_requirements.txt"
        TEST_OUTPUT = "data/output"

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    BATCH_SIZE = 64
    
    # Determinism Anchor (SPEC-2.2)
    TODAY = load_dataset_today()