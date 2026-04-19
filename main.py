"""
Pipeline Orchestrator
---------------------
Wires every component together and exposes a simple query() method.
"""

from __future__ import annotations
import logging
import time
import contextvars
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

from config import PipelineConfig
from models import (
    Document,
    PipelineResult,
    EvaluatorOutput,
    EvaluationStatus,
)
from llm_client import LLMClient
from embeddings import EmbeddingModel
from vector_store import VectorStore
from document_loader import DocumentLoader
from retriever import Retriever
from guardrail_agent import GuardrailAgent
from generator import Generator
from evaluator_agent import EvaluatorAgent
from telemetry import (
    configure_tracer_provider,
    get_or_create_correlation_id,
    get_tracer,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end multi-agent RAG pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()

        self._llm = LLMClient(self._config.llm)
        self._embeddings = EmbeddingModel(self._config.embedding)
        self._vector_store = VectorStore(
            self._config.vector_store,
            dimension=self._embeddings.dimension,
        )
        self._loader = DocumentLoader(self._config.chunking)
        self._retriever = Retriever(
            self._config.retriever, self._embeddings, self._vector_store
        )

        self._guardrail = GuardrailAgent(self._config.guardrail, self._llm)
        self._generator = Generator(self._config.generator, self._llm)
        self._evaluator = EvaluatorAgent(self._config.evaluator, self._llm)
        configure_tracer_provider(self._config.telemetry)
        self._tracer = get_tracer("rag.main")

        self._evaluator_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-evaluator")

        logger.info("RAG Pipeline initialized")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.exception("Failed to close RAG pipeline during finalization")

    def ingest(
        self,
        texts: list[str],
        source: str = "manual",
        metadata: Optional[dict] = None,
    ) -> int:
        docs = [
            Document(content=t, source=f"{source}_{i}", metadata=metadata or {})
            for i, t in enumerate(texts)
        ]
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: list[Document]) -> int:
        chunks = self._loader.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks produced from documents")
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embeddings.embed(texts)
        self._vector_store.add(chunks, embeddings)
        return len(chunks)

    def query(self, question: str) -> PipelineResult:
        t0 = time.perf_counter()
        correlation_id = get_or_create_correlation_id()
        tracer = self._tracer

        query_ctx = tracer.start_as_current_span("rag.query") if tracer else nullcontext()
        with query_ctx as query_span:
            if query_span:
                query_span.set_attribute("correlation_id", correlation_id)
                query_span.set_attribute("query_length", len(question))

            logger.info(
                "query.start event=query_start correlation_id=%s stage=query query_length=%s",
                correlation_id,
                len(question),
            )

            retrieve_ctx = tracer.start_as_current_span("rag.retrieve") if tracer else nullcontext()
            with retrieve_ctx as retrieve_span:
                retrieved = self._retriever.retrieve(question)
                if retrieve_span:
                    retrieve_span.set_attribute("retrieved_count", len(retrieved))
            logger.info("  Step 1 (Retrieve): %s chunks", len(retrieved))

            filtered = [
                r for r in retrieved
                if r.similarity_score >= self._config.retriever.similarity_threshold
            ]

            guardrail_output = None

            if self._config.runtime.use_guardrail:
                guardrail_output = self._guardrail.evaluate(question, filtered)
                filtered = guardrail_output.filtered_chunks
                logger.info("  Step 2 (Guardrail): %s/%s chunks kept", len(filtered), len(retrieved))
            else:
                logger.info("  Step 2 (Threshold Gate): %s/%s chunks kept", len(filtered), len(retrieved))

            generate_ctx = tracer.start_as_current_span("rag.generate") if tracer else nullcontext()
            with generate_ctx as generate_span:
                gen_output = self._generator.generate(question, filtered)
                if generate_span:
                    generate_span.set_attribute("answer_chars", len(gen_output.answer))
            logger.info("  Step 3 (Generate): %s chars", len(gen_output.answer))

            evaluation_status = EvaluationStatus.PENDING
            evaluation_error = None
            evaluation_deferred = self._config.runtime.evaluator_mode != "sync"

            placeholder_eval = EvaluatorOutput(
                overall_consistency_score=0.0,
                is_reliable=False,
                claims=[],
                summary="Evaluation scheduled asynchronously.",
                processing_time_ms=0.0,
            )
            eval_output = placeholder_eval

            if self._config.runtime.evaluator_mode == "sync":
                eval_output, evaluation_status, evaluation_error = self._evaluate_safe(
                    answer=gen_output.answer,
                    context_chunks=filtered,
                    query=question,
                )
                logger.info(
                    "  Step 4 (Evaluate sync): status=%s, score=%.2f",
                    evaluation_status.value,
                    eval_output.overall_consistency_score,
                )
            else:
                executor = self._evaluator_executor
                if executor is not None:
                    worker_context = contextvars.copy_context()
                    executor.submit(
                        worker_context.run,
                        self._evaluate_safe,
                        answer=gen_output.answer,
                        context_chunks=filtered,
                        query=question,
                    )
                    logger.info("  Step 4 (Evaluate deferred): scheduled")

            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "query.complete event=query_complete correlation_id=%s stage=query duration_ms=%.2f retrieved_count=%s filtered_count=%s",
                correlation_id,
                total_ms,
                len(retrieved),
                len(filtered),
            )

            if query_span:
                query_span.set_attribute("duration_ms", total_ms)

            return PipelineResult(
                query=question,
                answer=gen_output.answer,
                is_reliable=eval_output.is_reliable,
                consistency_score=eval_output.overall_consistency_score,
                retrieval=retrieved,
                guardrail=guardrail_output,
                generation=gen_output,
                evaluation=eval_output,
                evaluation_status=evaluation_status,
                evaluation_error=evaluation_error,
                evaluation_deferred=evaluation_deferred,
                total_time_ms=total_ms,
            )

    def _evaluate_safe(self, answer: str, context_chunks: list, query: str) -> Tuple[EvaluatorOutput, EvaluationStatus, Optional[str]]:
        try:
            evaluate_ctx = self._tracer.start_as_current_span("rag.evaluate") if self._tracer else nullcontext()
            with evaluate_ctx as evaluate_span:
                eval_output = self._evaluator.evaluate(
                    answer=answer,
                    context_chunks=context_chunks,
                    query=query,
                )
                if evaluate_span:
                    evaluate_span.set_attribute("evaluation_status", EvaluationStatus.COMPLETED.value)
            return eval_output, EvaluationStatus.COMPLETED, None
        except Exception as exc:
            logger.exception("Evaluator failed: %s", exc)
            fallback = EvaluatorOutput(
                overall_consistency_score=0.0,
                is_reliable=False,
                claims=[],
                summary="Evaluator failed; answer returned without blocking.",
                processing_time_ms=0.0,
            )
            return fallback, EvaluationStatus.FAILED, str(exc)

    def save(self, name: str = "default") -> None:
        self._vector_store.save(name)

    def load(self, name: str = "default") -> None:
        self._vector_store.load(name)

    def close(self) -> None:
        executor = getattr(self, "_evaluator_executor", None)
        if executor is None:
            return

        try:
            executor.shutdown(wait=True)
        except Exception:
            logger.exception("Failed to shut down evaluator executor")
        finally:
            self._evaluator_executor = None
