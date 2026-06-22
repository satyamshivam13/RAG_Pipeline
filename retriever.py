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
from telemetry import (
    add_counter,
    get_or_create_correlation_id,
    observe_duration,
    set_span_attributes,
    span_context_or_null,
)

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
        with span_context_or_null(
            "rag.retriever.retrieve",
            {"retriever.use_mmr": self._config.use_mmr, "retriever.top_k": self._config.top_k},
            "rag.retriever",
        ) as span:
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

            # Preserve MMR selection order; only similarity search needs sorting.
            if not self._config.use_mmr:
                results.sort(key=lambda r: r.similarity_score, reverse=True)

            elapsed = observe_duration(
                "rag_retrieval_latency_ms",
                t0,
                attributes={"use_mmr": self._config.use_mmr, "result_count": len(results)},
            )
            add_counter("rag_retrieval_requests_total", attributes={"use_mmr": self._config.use_mmr})
            set_span_attributes(span, {"retrieved.count": len(results), "duration_ms": elapsed})
        correlation_id = get_or_create_correlation_id()
        logger.info(
            "retriever.complete event=retrieve_done correlation_id=%s "
            "component=retriever operation=retrieve stage=retrieve "
            "duration_ms=%.2f retrieved_count=%s",
            correlation_id,
            elapsed,
            len(results),
        )
        return results
