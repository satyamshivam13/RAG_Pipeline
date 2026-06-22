"""
Embedding model wrapper.
"""

from __future__ import annotations
import logging
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig
from telemetry import add_counter, observe_duration, set_span_attributes, span_context_or_null

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

        t0 = time.perf_counter()
        with span_context_or_null(
            "rag.embedding.embed",
            {
                "embedding.model": self._config.model_name,
                "embedding.batch_size": self._config.batch_size,
                "embedding.text_count": len(texts),
            },
            "rag.embeddings",
        ) as span:
            embeddings = self._model.encode(
                texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
            elapsed_ms = observe_duration(
                "rag_embedding_latency_ms",
                t0,
                attributes={"model": self._config.model_name, "text_count": len(texts)},
            )
            add_counter("rag_embedding_batches_total", attributes={"model": self._config.model_name})
            set_span_attributes(span, {"duration_ms": elapsed_ms, "embedding.dimension": self._config.dimension})
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._config.dimension
