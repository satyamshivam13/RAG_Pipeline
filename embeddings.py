"""
Embedding model wrapper.  Isolates the sentence-transformers dependency so
swapping to OpenAI embeddings or a local ONNX model is a one-file change.
"""

from __future__ import annotations
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, config: EmbeddingConfig):
        self._config = config
        logger.info(f"Loading embedding model: {config.model_name}")
        self._model = SentenceTransformer(
            config.model_name,
            device=config.device,
        )
        # Validate dimension
        test_emb = self._model.encode(["test"], normalize_embeddings=config.normalize)
        actual_dim = test_emb.shape[1]
        if actual_dim != config.dimension:
            raise ValueError(
                f"Model outputs dim={actual_dim} but config says {config.dimension}. "
                f"Update EmbeddingConfig.dimension to {actual_dim}."
            )
        logger.info(f"Embedding model ready. dim={actual_dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings → (N, D) float32 numpy array.
        Handles batching internally.
        """
        if not texts:
            return np.empty((0, self._config.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → (D,) vector."""
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._config.dimension