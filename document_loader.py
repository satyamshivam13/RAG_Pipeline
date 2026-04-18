"""
Document ingestion: load text and split into semantic-aware overlapping chunks.
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

    def load_text(self, text: str, source: str = "inline") -> Document:
        return Document(content=text, source=source)

    def load_file(self, path: str | Path) -> Document:
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        return Document(content=content, source=str(path))

    def load_texts(self, texts: list[str], source: str = "batch") -> list[Document]:
        return [Document(content=t, source=f"{source}_{i}") for i, t in enumerate(texts)]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a document into overlapping chunks with semantic breakpoint preference."""
        text = doc.content
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._config.chunk_size
            actual_end = self._find_break_point(text, start, end)

            # Guard against non-progressing windows.
            if actual_end <= start:
                actual_end = min(len(text), start + self._config.chunk_size)
                if actual_end <= start:
                    break

            chunk_text = text[start:actual_end].strip()

            if len(chunk_text) >= self._config.min_chunk_size:
                chunks.append(
                    Chunk(
                        document_id=doc.id,
                        content=chunk_text,
                        source=doc.source,
                        chunk_index=idx,
                        metadata=doc.metadata.copy(),
                    )
                )
                idx += 1

            if actual_end >= len(text):
                break

            # Maintain backward-compatible overlap semantics.
            next_start = max(actual_end - self._config.chunk_overlap, start + 1)
            start = next_start

        logger.debug("Document %s: %s chunks", doc.source, len(chunks))
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Chunked %s docs to %s chunks", len(docs), len(all_chunks))
        return all_chunks

    @staticmethod
    def _find_break_point(text: str, start: int, end: int) -> int:
        """Choose a semantically meaningful break near end, fallback to hard split."""
        if end >= len(text):
            return len(text)

        search_region = text[start:end]

        # Paragraph boundaries first, then sentence boundaries.
        delimiters = ["\n\n", ".\n", "!\n", "?\n", ". ", "! ", "? ", "\n"]
        for delim in delimiters:
            last = search_region.rfind(delim)
            if last != -1 and last > len(search_region) * 0.3:
                return start + last + len(delim)

        # Fallback: split at nearest space before hard limit.
        space = search_region.rfind(" ")
        if space != -1 and space > len(search_region) * 0.3:
            return start + space + 1

        return end
