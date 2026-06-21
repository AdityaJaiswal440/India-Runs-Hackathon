# src/utils/logger.py
import logging
from pathlib import Path
import sys


def get_logger(stage_name: str) -> logging.Logger:
    """
    Returns a configured logger that saves to: logs/{stage_name}.log
    Automatically creates the /logs directory if it doesn't exist.
    """
    logger = logging.getLogger(stage_name)

    if not logger.handlers:
        # 1. Automatically build the 'logs' folder at the root of the project
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 2. Sanitize the name (in case someone passes "stage/1" with a slash)
        safe_name = stage_name.replace("/", "_").replace("\\", "_")
        log_file = log_dir / f"{safe_name}.log"

        # 3. Create Handlers
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    return logger