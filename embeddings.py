"""
Embedding model wrapper.
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
        logger.info("Loading embedding model: %s", config.model_name)
        self._model = SentenceTransformer(config.model_name, device=config.device)

        test_emb = self._model.encode(["test"], normalize_embeddings=config.normalize)
        actual_dim = int(test_emb.shape[1])
        if actual_dim != config.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model '{config.model_name}' outputs {actual_dim}, "
                f"but EmbeddingConfig.dimension is {config.dimension}. "
                "Update config dimension or rebuild index for matching vectors."
            )

        logger.info("Embedding model ready. dim=%s", actual_dim)

    def embed(self, texts: list[str]) -> np.ndarray:
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
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._config.dimension
