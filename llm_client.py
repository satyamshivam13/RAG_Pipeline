"""
Unified LLM client with retry logic, token counting, and structured output parsing.
Wraps the OpenAI SDK so the rest of the codebase never touches HTTP directly.
"""

from __future__ import annotations
import json
import logging
import re
import time
from typing import Any, Optional, cast

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from config import LLMConfig
from telemetry import (
    add_counter,
    get_or_create_correlation_id,
    observe_duration,
    set_span_attributes,
    span_context_or_null,
)

logger = logging.getLogger(__name__)
_ENCODER_CACHE: dict[str, Any] = {}
_ENCODER_FALLBACKS: set[str] = set()


class _RegexTokenCounter:
    _TOKEN_RE = re.compile(r"\s+|[^\s]+", re.UNICODE)

    def encode(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text)


class LLMClient:
    """Thread-safe, retry-aware LLM client."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._encoder: Any
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        # Token counter (falls back to an offline regex counter when tiktoken
        # cannot resolve or load an encoding in restricted environments).
        if config.default_model in _ENCODER_CACHE:
            self._encoder = _ENCODER_CACHE[config.default_model]
        elif config.default_model in _ENCODER_FALLBACKS:
            self._encoder = _RegexTokenCounter()
        else:
            try:
                self._encoder = tiktoken.encoding_for_model(config.default_model)
                _ENCODER_CACHE[config.default_model] = self._encoder
            except Exception:
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                    _ENCODER_CACHE[config.default_model] = self._encoder
                except Exception:
                    self._encoder = _RegexTokenCounter()
                    _ENCODER_FALLBACKS.add(config.default_model)

    # ── Public API ──────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request. Returns the assistant's text."""
        model = model or self._config.default_model

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        correlation_id = get_or_create_correlation_id()
        logger.info(
            "llm.request event=llm_request correlation_id=%s component=llm_client operation=chat model=%s messages=%s",
            correlation_id,
            model,
            len(messages),
        )
        t0 = time.perf_counter()
        with span_context_or_null(
            "rag.llm.chat",
            {
                "llm.provider": self._config.provider,
                "llm.model": model,
                "llm.message_count": len(messages),
                "llm.max_tokens": max_tokens,
            },
            "rag.llm",
        ) as span:
            create = cast(Any, self._client.chat.completions.create)
            response = create(**kwargs)
            text = response.choices[0].message.content.strip()
            elapsed_ms = observe_duration(
                "rag_llm_latency_ms",
                t0,
                attributes={"provider": self._config.provider, "model": model},
            )
            add_counter(
                "rag_llm_requests_total",
                attributes={"provider": self._config.provider, "model": model, "status": "success"},
            )
            set_span_attributes(span, {"duration_ms": elapsed_ms, "llm.response_chars": len(text)})
        logger.info(
            "llm.response event=llm_response correlation_id=%s component=llm_client operation=chat chars=%s",
            correlation_id,
            len(text),
        )
        return text

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """Chat completion that returns parsed JSON."""
        raw = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return self._parse_json(raw)

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Robustly parse JSON, handling markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip ```json ... ```
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw: {raw[:500]}")
            raise
