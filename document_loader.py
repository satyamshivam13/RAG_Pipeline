"""
Document ingestion: load text and split into semantic-aware overlapping chunks.
Supports plain strings, .txt files, and lists of strings.
"""

from __future__ import annotations
import logging
from pathlib import Path

from chunking import ChunkPlan, build_chunk_strategy
from config import ChunkingConfig
from models import Document, Chunk

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self, config: ChunkingConfig):
        self._config = config
        self._chunker = build_chunk_strategy(config)

    def load_text(self, text: str, source: str = "inline") -> Document:
        return Document(content=text, source=source)

    def load_file(self, path: str | Path) -> Document:
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        return Document(content=content, source=str(path))

    def load_texts(self, texts: list[str], source: str = "batch") -> list[Document]:
        return [Document(content=t, source=f"{source}_{i}") for i, t in enumerate(texts)]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split a document into configured token-aware chunks."""
        chunks: list[Chunk] = []
        plans = self._chunker.split(doc.content)

        for idx, plan in enumerate(plans):
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    content=plan.content,
                    source=doc.source,
                    chunk_index=idx,
                    metadata=self._metadata_with_chunk_quality(doc, plan),
                )
            )

        logger.debug("Document %s: %s chunks", doc.source, len(chunks))
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Chunked %s docs to %s chunks", len(docs), len(all_chunks))
        return all_chunks

    def _metadata_with_chunk_quality(self, doc: Document, plan: ChunkPlan) -> dict:
        metadata = doc.metadata.copy()
        metadata["chunking"] = {
            "strategy": self._config.strategy,
            "tokenizer_model": self._config.tokenizer_model,
            "metrics": plan.metrics.as_dict(),
        }
        return metadata
