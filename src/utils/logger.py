# src/utils/logger.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with a standard format.
    Using a function prevents duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger