"""
Pipeline Orchestrator
─────────────────────
Wires every component together and exposes a simple `query()` method.

Flow:
  1. Retriever fetches top-k chunks from the vector store.
  2. Guardrail Agent filters for relevance & safety.
  3. Generator produces an answer from the filtered context.
  4. Evaluator Agent scores the answer for factual consistency.
  5. Everything is packaged into a PipelineResult.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

from config import PipelineConfig
from models import Document, Chunk, PipelineResult
from llm_client import LLMClient
from embeddings import EmbeddingModel
from vector_store import VectorStore
from document_loader import DocumentLoader
from retriever import Retriever
from guardrail_agent import GuardrailAgent
from generator import Generator
from evaluator_agent import EvaluatorAgent

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end multi-agent RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(["doc1 text", "doc2 text"])
        result = pipeline.query("What is X?")
        print(result.answer)
        print(result.consistency_score)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()

        # ── Shared components ───────────────────────────────────────
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

        # ── Agents ──────────────────────────────────────────────────
        self._guardrail = GuardrailAgent(self._config.guardrail, self._llm)
        self._generator = Generator(self._config.generator, self._llm)
        self._evaluator = EvaluatorAgent(self._config.evaluator, self._llm)

        logger.info("RAG Pipeline initialized")

    # ── Ingestion ───────────────────────────────────────────────────

    def ingest(
        self,
        texts: list[str],
        source: str = "manual",
        metadata: Optional[dict] = None,
    ) -> int:
        """Ingest raw texts into the vector store. Returns chunk count."""
        docs = [
            Document(content=t, source=f"{source}_{i}", metadata=metadata or {})
            for i, t in enumerate(texts)
        ]
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Ingest Document objects: chunk → embed → store."""
        chunks = self._loader.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks produced from documents")
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embeddings.embed(texts)
        self._vector_store.add(chunks, embeddings)
        return len(chunks)

    # ── Query ───────────────────────────────────────────────────────

    def query(self, question: str) -> PipelineResult:
        """
        Run the full pipeline:
          retrieve → guardrail → generate → evaluate → return
        """
        t0 = time.perf_counter()
        logger.info(f"Pipeline query: '{question[:80]}...'")

        # ── Step 1: Retrieve ────────────────────────────────────────
        retrieved = self._retriever.retrieve(question)
        logger.info(f"  Step 1 (Retrieve): {len(retrieved)} chunks")

        # ── Step 2: Guardrail ───────────────────────────────────────
        guardrail_output = self._guardrail.evaluate(question, retrieved)
        filtered = guardrail_output.filtered_chunks
        logger.info(
            f"  Step 2 (Guardrail): {len(filtered)}/{len(retrieved)} chunks kept"
        )

        # ── Step 3: Generate ────────────────────────────────────────
        gen_output = self._generator.generate(question, filtered)
        logger.info(f"  Step 3 (Generate): {len(gen_output.answer)} chars")

        # ── Step 4: Evaluate ────────────────────────────────────────
        eval_output = self._evaluator.evaluate(
            answer=gen_output.answer,
            context_chunks=filtered,
            query=question,
        )
        logger.info(
            f"  Step 4 (Evaluate): score={eval_output.overall_consistency_score:.2f}"
        )

        total_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            query=question,
            answer=gen_output.answer,
            is_reliable=eval_output.is_reliable,
            consistency_score=eval_output.overall_consistency_score,
            retrieval=retrieved,
            guardrail=guardrail_output,
            generation=gen_output,
            evaluation=eval_output,
            total_time_ms=total_ms,
        )

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, name: str = "default") -> None:
        self._vector_store.save(name)

    def load(self, name: str = "default") -> None:
        self._vector_store.load(name)