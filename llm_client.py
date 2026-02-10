"""
Unified LLM client with retry logic, token counting, and structured output parsing.
Wraps the OpenAI SDK so the rest of the codebase never touches HTTP directly.
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tiktoken

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """Thread-safe, retry-aware LLM client."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        # Token counter (falls back to cl100k_base for unknown models)
        try:
            self._encoder = tiktoken.encoding_for_model(config.default_model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

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

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"LLM request: model={model}, msgs={len(messages)}")
        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content.strip()
        logger.debug(f"LLM response: {len(text)} chars")
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