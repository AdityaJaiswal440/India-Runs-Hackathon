# src/data/loader.py
import gzip
import json
import os
import shutil
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_data(filepath: str) -> str:
    """
    Ensures we have a decompressed .jsonl file to work with.
    Returns the path to the usable .jsonl file.
    """
    # If the user provides a .jsonl file, assume it's ready to go.
    if filepath.endswith('.jsonl'):
        if os.path.exists(filepath):
            logger.info(f"Using ready JSONL file: {filepath}")
            return filepath
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

    # If the user provides a .gz file, we need to check for a decompressed version
    elif filepath.endswith('.jsonl.gz'):
        # Define where the decompressed file should go (in data/interim/)
        filename = os.path.basename(filepath).replace('.gz', '')
        interim_dir = "data/interim"
        os.makedirs(interim_dir, exist_ok=True)
        decompressed_path = os.path.join(interim_dir, filename)

        # 1. Check if user already hand-decompressed it in the raw folder
        raw_decompressed = filepath.replace('.gz', '')
        if os.path.exists(raw_decompressed):
            logger.info(f"Found hand-decompressed file at {raw_decompressed}. Using it.")
            return raw_decompressed

        # 2. Check if we already decompressed it in the interim folder previously
        if os.path.exists(decompressed_path):
            logger.info(f"Found previously decompressed file at {decompressed_path}. Using it.")
            return decompressed_path

        # 3. If neither exists, decompress it now
        if os.path.exists(filepath):
            logger.info(f"Decompressing {filepath} to {decompressed_path}...")
            with gzip.open(filepath, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info("Decompression complete.")
            return decompressed_path
        else:
            raise FileNotFoundError(f"File not found: {filepath}")
    
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .jsonl.gz file.")

def stream_candidates(filepath: str):
    """
    Generator that streams candidates from a .jsonl file.
    """
    # Ensure the file is ready and get the usable path
    usable_path = prepare_data(filepath)
    
    logger.info(f"Starting to stream candidates from: {usable_path}")
    
    try:
        with open(usable_path, 'rt', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    candidate = json.loads(line)
                    
                    # Defensive validation
                    if "candidate_id" not in candidate or "redrob_signals" not in candidate:
                        logger.warning(f"Skipping malformed candidate at line {count+1}: missing core fields")
                        continue
                    
                    yield candidate
                    count += 1
                    
                    if count % 10000 == 0:
                        logger.info(f"Streamed {count} candidates...")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {count+1}")
                    continue
                    
            logger.info(f"Successfully finished streaming {count} total candidates.")
            
    except FileNotFoundError:
        logger.error(f"File not found: {usable_path}")
        raise