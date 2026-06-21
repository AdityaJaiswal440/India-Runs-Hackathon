# src/features/honeypot.py
from src.utils.logger import get_logger

logger = get_logger(__name__)

def is_honeypot(candidate: dict) -> bool:
    """
    STUB FUNCTION: Teammate to implement actual logic.
    Currently returns False so pipeline can run end-to-end.
    """
    return False