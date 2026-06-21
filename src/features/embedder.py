# src/features/embedder.py
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
        # We load the model once and keep it in memory
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        
    def embed_texts(self, texts: list) -> np.ndarray:
        """
        Embeds a list of strings and returns a numpy array of shape (len(texts), 384).
        """
        if not texts:
            return np.array([])
            
        # show_progress_bar=False to keep logs clean during batch processing
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False,
            batch_size=Config.BATCH_SIZE
        )
        return embeddings