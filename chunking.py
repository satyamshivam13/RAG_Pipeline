"""
Token-aware document chunking strategies for retrieval.

The default strategy preserves paragraph and sentence boundaries where possible,
then enforces a hard token budget so downstream context packing stays predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

import tiktoken

from config import ChunkingConfig

_SENTENCE_ENDINGS = ".!?\u3002\uff01\uff1f\u0964\u061f"
_SENTENCE_RE = re.compile(rf".+?(?:[{re.escape(_SENTENCE_ENDINGS)}]+(?=\s|$)|$)", re.DOTALL)
_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
_TOKENIZER_FALLBACKS: set[tuple[str, str]] = set()


class Tokenizer(Protocol):
    def encode(self, text: str) -> list: ...

    def decode(self, tokens: list) -> str: ...

    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    """Small adapter around tiktoken with a stable fallback encoding."""

    def __init__(self, model: str = "gpt-4o-mini", encoding_name: str = "cl100k_base"):
        key = (model, encoding_name)
        self._fallback: RegexTokenizer | None = None
        # Holds a tiktoken Encoding (untyped third-party) or None when a fallback is used.
        self._encoding: Any = None
        if encoding_name == "regex":
            self._encoding = None
            self._fallback = RegexTokenizer()
            return
        if key in _TOKENIZER_CACHE:
            self._encoding = _TOKENIZER_CACHE[key]
            return
        if key in _TOKENIZER_FALLBACKS:
            self._encoding = None
            self._fallback = RegexTokenizer()
            return

        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
            _TOKENIZER_CACHE[key] = self._encoding
        except Exception:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
                _TOKENIZER_CACHE[key] = self._encoding
            except Exception:
                self._encoding = None
                self._fallback = RegexTokenizer()
                _TOKENIZER_FALLBACKS.add(key)

    def encode(self, text: str) -> list:
        if self._fallback is not None:
            return self._fallback.encode(text)
        return self._encoding.encode(text)

    def decode(self, tokens: list) -> str:
        if self._fallback is not None:
            return self._fallback.decode(tokens)
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


class RegexTokenizer:
    """Offline tokenizer fallback that preserves text during decode."""

    _TOKEN_RE = re.compile(r"\s+|[^\s]+", re.UNICODE)

    def encode(self, text: str) -> list[str]:
        return self._TOKEN_RE.findall(text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


@dataclass(frozen=True)
class ChunkQualityMetrics:
    token_count: int
    char_count: int
    sentence_count: int
    paragraph_count: int
    overlap_tokens: int
    token_budget_utilization: float
    starts_at_semantic_boundary: bool
    ends_at_semantic_boundary: bool
    oversized: bool = False

    def as_dict(self) -> dict:
        return {
            "token_count": self.token_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "overlap_tokens": self.overlap_tokens,
            "token_budget_utilization": self.token_budget_utilization,
            "starts_at_semantic_boundary": self.starts_at_semantic_boundary,
            "ends_at_semantic_boundary": self.ends_at_semantic_boundary,
            "oversized": self.oversized,
        }


@dataclass(frozen=True)
class ChunkPlan:
    content: str
    metrics: ChunkQualityMetrics


class ChunkStrategy(Protocol):
    name: str

    def split(self, text: str) -> list[ChunkPlan]: ...


@dataclass(frozen=True)
class _TextUnit:
    text: str
    separator_before: str = ""


class SemanticTokenChunker:
    """Paragraph-first, sentence-aware chunker with token-budget guarantees."""

    name = "semantic"

    def __init__(self, config: ChunkingConfig, tokenizer: Tokenizer | None = None):
        self._config = config
        self._tokenizer = tokenizer or TiktokenTokenizer(
            model=config.tokenizer_model,
            encoding_name=config.tokenizer_encoding,
        )
        self._chunk_size = max(1, config.chunk_size)
        self._overlap = max(0, min(config.chunk_overlap, self._chunk_size - 1))

    def split(self, text: str) -> list[ChunkPlan]:
        if not text or not text.strip():
            return []

        units = self._semantic_units(text)
        raw_chunks = self._pack_units(units)
        raw_chunks = self._merge_or_drop_tiny_tail(raw_chunks)
        return self._with_overlap(raw_chunks)

    def _semantic_units(self, text: str) -> list[_TextUnit]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        units: list[_TextUnit] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_separator = "\n\n" if paragraph_index > 0 else ""
            if self._tokenizer.count(paragraph) <= self._chunk_size:
                units.append(_TextUnit(paragraph, paragraph_separator))
                continue

            sentences = self._split_sentences(paragraph)
            for sentence_index, sentence in enumerate(sentences):
                separator = paragraph_separator if sentence_index == 0 else " "
                if self._tokenizer.count(sentence) <= self._chunk_size:
                    units.append(_TextUnit(sentence, separator))
                    continue

                for piece_index, piece in enumerate(self._split_by_tokens(sentence)):
                    piece_separator = separator if piece_index == 0 else " "
                    units.append(_TextUnit(piece, piece_separator))

        return units

    def _pack_units(self, units: list[_TextUnit]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for unit in units:
            candidate = self._append_unit(current, unit)
            if current and self._tokenizer.count(candidate) > self._chunk_size:
                chunks.append(current.strip())
                current = unit.text.strip()
            else:
                current = candidate.strip()

        if current:
            chunks.append(current.strip())

        return chunks

    def _with_overlap(self, chunks: list[str]) -> list[ChunkPlan]:
        plans: list[ChunkPlan] = []
        previous = ""

        for index, chunk in enumerate(chunks):
            overlap_tokens = 0
            content = chunk
            if index > 0 and self._overlap > 0 and previous:
                previous_tokens = self._tokenizer.encode(previous)
                max_overlap = min(self._overlap, len(previous_tokens))
                for allowed_overlap in range(max_overlap, 0, -1):
                    overlap_text = self._tokenizer.decode(previous_tokens[-allowed_overlap:]).strip()
                    if overlap_text:
                        candidate = f"{overlap_text}\n\n{chunk}".strip()
                        if self._tokenizer.count(candidate) <= self._chunk_size:
                            content = candidate
                            overlap_tokens = self._tokenizer.count(overlap_text)
                            break

            plans.append(self._plan(content, overlap_tokens))
            previous = chunk

        return plans

    def _merge_or_drop_tiny_tail(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        min_tokens = max(0, self._config.min_chunk_size)
        if min_tokens == 0 or self._tokenizer.count(chunks[-1]) >= min_tokens:
            return chunks

        candidate = f"{chunks[-2]}\n\n{chunks[-1]}".strip()
        if self._tokenizer.count(candidate) <= self._chunk_size:
            return chunks[:-2] + [candidate]

        return chunks[:-1]

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self._tokenizer.encode(text)
        if not tokens:
            return []

        pieces: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self._chunk_size)
            pieces.append(self._tokenizer.decode(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start = end
        return [piece for piece in pieces if piece]

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
        return [sentence for sentence in sentences if sentence] or [text.strip()]

    def _append_unit(self, current: str, unit: _TextUnit) -> str:
        if not current:
            return unit.text.strip()
        separator = unit.separator_before or " "
        return f"{current}{separator}{unit.text.strip()}"

    def _plan(self, content: str, overlap_tokens: int = 0) -> ChunkPlan:
        token_count = self._tokenizer.count(content)
        metrics = ChunkQualityMetrics(
            token_count=token_count,
            char_count=len(content),
            sentence_count=len(self._split_sentences(content)),
            paragraph_count=len([p for p in re.split(r"\n\s*\n", content) if p.strip()]),
            overlap_tokens=overlap_tokens,
            token_budget_utilization=round(token_count / self._chunk_size, 4),
            starts_at_semantic_boundary=self._starts_at_semantic_boundary(content),
            ends_at_semantic_boundary=self._ends_at_semantic_boundary(content),
            oversized=token_count > self._chunk_size,
        )
        return ChunkPlan(content=content, metrics=metrics)

    @staticmethod
    def _starts_at_semantic_boundary(text: str) -> bool:
        return bool(text) and not text[0].islower()

    @staticmethod
    def _ends_at_semantic_boundary(text: str) -> bool:
        return bool(text.rstrip()) and text.rstrip()[-1] in _SENTENCE_ENDINGS


class TokenWindowChunker(SemanticTokenChunker):
    """Fast token-window strategy for callers that do not want semantic packing."""

    name = "token"

    def split(self, text: str) -> list[ChunkPlan]:
        if not text or not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        chunks: list[str] = []
        start = 0
        step = max(1, self._chunk_size - self._overlap)

        while start < len(tokens):
            end = min(len(tokens), start + self._chunk_size)
            chunks.append(self._tokenizer.decode(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start += step

        chunks = self._merge_or_drop_tiny_tail([chunk for chunk in chunks if chunk])
        return [self._plan(chunk, self._overlap if index > 0 else 0) for index, chunk in enumerate(chunks)]


class LegacyCharacterChunker:
    """Backward-compatible character strategy retained for benchmarks and rollbacks."""

    name = "legacy_char"

    def __init__(self, config: ChunkingConfig, tokenizer: Tokenizer | None = None):
        self._config = config
        self._tokenizer = tokenizer or TiktokenTokenizer(
            model=config.tokenizer_model,
            encoding_name=config.tokenizer_encoding,
        )

    def split(self, text: str) -> list[ChunkPlan]:
        chunks: list[ChunkPlan] = []
        start = 0

        while start < len(text):
            end = start + self._config.chunk_size
            actual_end = self._find_break_point(text, start, end)
            if actual_end <= start:
                actual_end = min(len(text), start + self._config.chunk_size)
                if actual_end <= start:
                    break

            chunk_text = text[start:actual_end].strip()
            if len(chunk_text) >= self._config.min_chunk_size or not chunks:
                overlap_tokens = 0
                if chunks and self._config.chunk_overlap > 0:
                    overlap_tokens = self._tokenizer.count(chunk_text[: self._config.chunk_overlap])
                chunks.append(self._plan(chunk_text, overlap_tokens=overlap_tokens))

            if actual_end >= len(text):
                break

            start = max(actual_end - self._config.chunk_overlap, start + 1)

        return chunks

    def _plan(self, content: str, overlap_tokens: int = 0) -> ChunkPlan:
        token_count = self._tokenizer.count(content)
        sentence_count = len(
            [match.group(0).strip() for match in _SENTENCE_RE.finditer(content) if match.group(0).strip()]
        )
        paragraph_count = len([p for p in re.split(r"\n\s*\n", content) if p.strip()])
        metrics = ChunkQualityMetrics(
            token_count=token_count,
            char_count=len(content),
            sentence_count=max(1, sentence_count) if content else 0,
            paragraph_count=paragraph_count,
            overlap_tokens=overlap_tokens,
            token_budget_utilization=round(token_count / max(1, self._config.chunk_size), 4),
            starts_at_semantic_boundary=SemanticTokenChunker._starts_at_semantic_boundary(content),
            ends_at_semantic_boundary=SemanticTokenChunker._ends_at_semantic_boundary(content),
            oversized=token_count > self._config.chunk_size,
        )
        return ChunkPlan(content=content, metrics=metrics)

    @staticmethod
    def _find_break_point(text: str, start: int, end: int) -> int:
        if end >= len(text):
            return len(text)

        search_region = text[start:end]
        delimiters = ["\n\n", ".\n", "!\n", "?\n", ". ", "! ", "? ", "\n"]
        for delimiter in delimiters:
            last = search_region.rfind(delimiter)
            if last != -1 and last > len(search_region) * 0.3:
                return start + last + len(delimiter)

        space = search_region.rfind(" ")
        if space != -1 and space > len(search_region) * 0.3:
            return start + space + 1

        return end


def build_chunk_strategy(config: ChunkingConfig, tokenizer: Tokenizer | None = None) -> ChunkStrategy:
    if config.strategy == "semantic":
        return SemanticTokenChunker(config, tokenizer)
    if config.strategy == "token":
        return TokenWindowChunker(config, tokenizer)
    if config.strategy == "legacy_char":
        return LegacyCharacterChunker(config, tokenizer)
    raise ValueError(f"Unsupported chunking strategy: {config.strategy}")
