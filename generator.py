"""
Generator Agent (LLM #3)
────────────────────────
Takes the query + guardrail-filtered context and produces a grounded answer.
Key design decisions:
  • Context is injected as numbered references so the model can cite them.
  • System prompt enforces "answer only from context" behavior.
  • If no context survives the guardrail, it says so explicitly.
"""

from __future__ import annotations
import logging
import time

from config import GeneratorConfig
from llm_client import LLMClient
from models import RetrievedChunk, GeneratorOutput

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, config: GeneratorConfig, llm: LLMClient):
        self._config = config
        self._llm = llm

    def generate(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> GeneratorOutput:
        t0 = time.perf_counter()

        # ── Build messages ──────────────────────────────────────────
        if not context_chunks:
            answer = (
                "I don't have enough relevant information in the provided "
                "knowledge base to answer this question reliably."
            )
            return GeneratorOutput(
                answer=answer,
                query=query,
                context_used=[],
                model=self._config.model,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

        context_block = self._build_context(context_chunks)
        user_prompt = (
            f"CONTEXT (use ONLY this to answer):\n"
            f"{context_block}\n\n"
            f"QUESTION:\n{query}\n\n"
            f"Provide a thorough, well-structured answer. "
            f"Cite context references as [1], [2], etc. where appropriate."
        )

        messages = [
            {"role": "system", "content": self._config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ── Call LLM ────────────────────────────────────────────────
        answer = self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Generator produced {len(answer)} chars in {elapsed:.0f}ms")

        return GeneratorOutput(
            answer=answer,
            query=query,
            context_used=context_chunks,
            model=self._config.model,
            processing_time_ms=elapsed,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] (source: {c.chunk.source}, "
                f"relevance: {c.similarity_score:.2f})\n"
                f"{c.chunk.content}"
            )
        return "\n\n".join(parts)