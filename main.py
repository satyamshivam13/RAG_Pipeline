"""
Pipeline Orchestrator
---------------------
Wires every component together and exposes a simple query() method.
"""

from __future__ import annotations
import logging
import time
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple, cast

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
    add_counter,
    configure_observability,
    get_or_create_correlation_id,
    observe_duration,
    record_histogram,
    set_span_attributes,
    span_context_or_null,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end multi-agent RAG pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()

        self._llm = LLMClient(self._config.llm)
        self._embeddings = EmbeddingModel(self._config.embedding)
        self._vector_store = VectorStore(
            cast(Any, self._config.vector_store),
            dimension=self._embeddings.dimension,
        )
        self._loader = DocumentLoader(self._config.chunking)
        self._retriever = Retriever(
            self._config.retriever, self._embeddings, self._vector_store
        )

        self._guardrail = GuardrailAgent(self._config.guardrail, self._llm)
        self._generator = Generator(self._config.generator, self._llm)
        self._evaluator = EvaluatorAgent(self._config.evaluator, self._llm)
        configure_observability(self._config.telemetry)

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
        t0 = time.perf_counter()
        with span_context_or_null("rag.ingest", {"document.count": len(documents)}, "rag.main") as span:
            chunks = self._loader.chunk_documents(documents)
            if not chunks:
                logger.warning("No chunks produced from documents")
                add_counter("rag_ingest_requests_total", attributes={"status": "empty"})
                return 0

            texts = [c.content for c in chunks]
            embeddings = self._embeddings.embed(texts)
            self._vector_store.add(chunks, embeddings)
            elapsed_ms = observe_duration(
                "rag_ingest_latency_ms",
                t0,
                attributes={"chunk_count": len(chunks), "document_count": len(documents)},
            )
            add_counter("rag_ingest_requests_total", attributes={"status": "success"})
            set_span_attributes(span, {"chunk.count": len(chunks), "duration_ms": elapsed_ms})
            return len(chunks)

    def query(
        self,
        question: str,
        enable_guardrail: Optional[bool] = None,
        sync_evaluation: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> PipelineResult:
        """Run a question through the pipeline.

        ``enable_guardrail``, ``sync_evaluation`` and ``top_k`` are optional
        per-request overrides (used by the API). When left as ``None`` the
        pipeline falls back to the configured runtime/retriever defaults, so
        existing callers that pass only ``question`` keep their behavior.
        """
        t0 = time.perf_counter()
        correlation_id = get_or_create_correlation_id()

        use_guardrail = (
            self._config.runtime.use_guardrail if enable_guardrail is None else enable_guardrail
        )
        run_sync_eval = (
            (self._config.runtime.evaluator_mode == "sync")
            if sync_evaluation is None
            else sync_evaluation
        )

        with span_context_or_null("rag.query", {"query.length": len(question)}, "rag.main") as query_span:
            if query_span:
                set_span_attributes(query_span, {"correlation_id": correlation_id})

            logger.info(
                "query.start event=query_start correlation_id=%s stage=query query_length=%s",
                correlation_id,
                len(question),
            )

            with span_context_or_null("rag.retrieve", tracer_name="rag.main") as retrieve_span:
                retrieved = self._retriever.retrieve(question, top_k=top_k)
                set_span_attributes(retrieve_span, {"retrieved.count": len(retrieved)})
            logger.info("  Step 1 (Retrieve): %s chunks", len(retrieved))

            filtered = [
                r for r in retrieved
                if r.similarity_score >= self._config.retriever.similarity_threshold
            ]

            guardrail_output = None

            if use_guardrail:
                guardrail_output = self._guardrail.evaluate(question, filtered)
                filtered = guardrail_output.filtered_chunks
                logger.info("  Step 2 (Guardrail): %s/%s chunks kept", len(filtered), len(retrieved))
            else:
                logger.info("  Step 2 (Threshold Gate): %s/%s chunks kept", len(filtered), len(retrieved))

            with span_context_or_null("rag.generate", tracer_name="rag.main") as generate_span:
                gen_output = self._generator.generate(question, filtered)
                set_span_attributes(generate_span, {"answer.chars": len(gen_output.answer)})
            logger.info("  Step 3 (Generate): %s chars", len(gen_output.answer))

            evaluation_status = EvaluationStatus.PENDING
            evaluation_error = None
            evaluation_deferred = not run_sync_eval

            placeholder_eval = EvaluatorOutput(
                overall_consistency_score=0.0,
                is_reliable=False,
                claims=[],
                summary="Evaluation scheduled asynchronously.",
                processing_time_ms=0.0,
            )
            eval_output = placeholder_eval

            if run_sync_eval:
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
            record_histogram(
                "rag_query_latency_ms",
                total_ms,
                attributes={
                    "retrieved_count": len(retrieved),
                    "filtered_count": len(filtered),
                    "evaluator_mode": self._config.runtime.evaluator_mode,
                },
            )
            add_counter("rag_query_requests_total", attributes={"status": "success"})
            logger.info(
                "query.complete event=query_complete correlation_id=%s "
                "stage=query duration_ms=%.2f retrieved_count=%s filtered_count=%s",
                correlation_id,
                total_ms,
                len(retrieved),
                len(filtered),
            )

            set_span_attributes(query_span, {"duration_ms": total_ms})

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

    def _evaluate_safe(
        self,
        answer: str,
        context_chunks: list,
        query: str,
    ) -> Tuple[EvaluatorOutput, EvaluationStatus, Optional[str]]:
        try:
            with span_context_or_null("rag.evaluate", tracer_name="rag.main") as evaluate_span:
                eval_output = self._evaluator.evaluate(
                    answer=answer,
                    context_chunks=context_chunks,
                    query=query,
                )
                set_span_attributes(evaluate_span, {"evaluation.status": EvaluationStatus.COMPLETED.value})
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
