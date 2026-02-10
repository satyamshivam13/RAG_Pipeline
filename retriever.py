"""
Retriever: thin orchestration layer over EmbeddingModel + VectorStore.
Handles the embed-query → search → rank workflow.
"""

from __future__ import annotations
import logging
import time

from config import RetrieverConfig
from embeddings import EmbeddingModel
from vector_store import VectorStore
from models import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        config: RetrieverConfig,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self._config = config
        self._embeddings = embedding_model
        self._store = vector_store

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Embed the query, search the vector store, optionally apply MMR.
        Returns chunks sorted by descending relevance.
        """
        t0 = time.perf_counter()
        query_vec = self._embeddings.embed_query(query)

        if self._config.use_mmr:
            results = self._store.mmr_search(
                query_embedding=query_vec,
                top_k=self._config.mmr_top_k,
                fetch_k=self._config.top_k,
                lambda_mult=self._config.mmr_lambda,
                threshold=self._config.similarity_threshold,
            )
        else:
            results = self._store.search(
                query_embedding=query_vec,
                top_k=self._config.top_k,
                threshold=self._config.similarity_threshold,
            )

        # Sort descending by score
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Retrieved {len(results)} chunks for query "
            f"'{query[:60]}...' in {elapsed:.1f}ms"
        )
        return results