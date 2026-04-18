"""
Generator Agent (LLM #3)

Takes the query plus retrieved context and produces a grounded answer.
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

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> GeneratorOutput:
        t0 = time.perf_counter()

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
                context_token_estimate=0,
            )

        selected_chunks, token_estimate, truncated_count = self._select_context_with_budget(context_chunks)
        context_block = self._build_context(selected_chunks)

        warnings = []
        if truncated_count > 0:
            warning = (
                f"Context truncated: dropped {truncated_count} chunk(s) to fit "
                f"max_context_tokens={self._config.max_context_tokens}."
            )
            warnings.append(warning)
            logger.warning(warning)

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

        answer = self._llm.chat(
            messages=messages,
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Generator produced %s chars in %.0fms", len(answer), elapsed)

        return GeneratorOutput(
            answer=answer,
            query=query,
            context_used=selected_chunks,
            model=self._config.model,
            processing_time_ms=elapsed,
            context_truncated=truncated_count > 0,
            warnings=warnings,
            context_token_estimate=token_estimate,
        )

    def _select_context_with_budget(self, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], int, int]:
        # Keep highest-relevance chunks first so lowest relevance are truncated first.
        ordered = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
        selected: list[RetrievedChunk] = []
        token_total = 0

        for chunk in ordered:
            chunk_tokens = self._estimate_tokens(chunk.chunk.content)
            if selected and token_total + chunk_tokens > self._config.max_context_tokens:
                continue
            if not selected and chunk_tokens > self._config.max_context_tokens:
                # Ensure at least one chunk remains to avoid empty-context hallucination path.
                selected.append(chunk)
                token_total = min(chunk_tokens, self._config.max_context_tokens)
                break

            selected.append(chunk)
            token_total += chunk_tokens

        truncated_count = max(0, len(chunks) - len(selected))
        return selected, token_total, truncated_count

    def _estimate_tokens(self, text: str) -> int:
        if hasattr(self._llm, "count_tokens"):
            try:
                return int(self._llm.count_tokens(text))
            except Exception:
                pass
        # Deterministic fallback heuristic.
        return max(1, len(text) // 4)

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] (source: {c.chunk.source}, relevance: {c.similarity_score:.2f})\n"
                f"{c.chunk.content}"
            )
        return "\n\n".join(parts)
