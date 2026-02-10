"""
Document ingestion: load text → split into overlapping chunks.
Supports plain strings, .txt files, and lists of strings.
"""

from __future__ import annotations
import logging
from pathlib import Path

from config import ChunkingConfig
from models import Document, Chunk

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self, config: ChunkingConfig):
        self._config = config

    # ── Public API ──────────────────────────────────────────────────

    def load_text(self, text: str, source: str = "inline") -> Document:
        return Document(content=text, source=source)

    def load_file(self, path: str | Path) -> Document:
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        return Document(content=content, source=str(path))

    def load_texts(self, texts: list[str], source: str = "batch") -> list[Document]:
        return [
            Document(content=t, source=f"{source}_{i}")
            for i, t in enumerate(texts)
        ]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a document into overlapping fixed-size chunks."""
        text = doc.content
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._config.chunk_size

            # Try to break at sentence boundary
            actual_end = self._find_break_point(text, start, end)
            chunk_text = text[start:actual_end].strip()

            if len(chunk_text) >= self._config.min_chunk_size:
                chunks.append(Chunk(
                    document_id=doc.id,
                    content=chunk_text,
                    source=doc.source,
                    chunk_index=idx,
                    metadata=doc.metadata.copy(),
                ))
                idx += 1

            start = actual_end - self._config.chunk_overlap
            if start >= len(text) or actual_end >= len(text):
                break

        logger.debug(f"Document {doc.source}: {len(chunks)} chunks")
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Chunked {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _find_break_point(text: str, start: int, end: int) -> int:
        """Try to break at the last sentence-ending punctuation before `end`."""
        if end >= len(text):
            return len(text)

        # Search backward from `end` for sentence boundaries
        search_region = text[start:end]
        for delim in [". ", ".\n", "!\n", "?\n", "! ", "? ", "\n\n"]:
            last = search_region.rfind(delim)
            if last != -1 and last > len(search_region) * 0.3:
                return start + last + len(delim)
        return end