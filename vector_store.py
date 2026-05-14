"""
FAISS-backed vector store with:
  - Insert / batch-insert
  - Similarity search (cosine / IP / L2)
  - Maximal Marginal Relevance (MMR)
  - Persistence to disk
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np

from config import VectorStoreConfig
from models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class _Candidate(NamedTuple):
    chunk: Chunk
    raw_score: float
    faiss_index: int


class VectorStore:
    def __init__(self, config: VectorStoreConfig, dimension: int):
        self._config = config
        self._dimension = dimension
        self._index = self._build_index()
        self._chunks: list[Chunk] = []
        # Store embeddings alongside chunks to avoid faiss.reconstruct()
        # This improves compatibility with IVF, PQ and other index types.
        self._embeddings: np.ndarray | None = np.zeros((0, self._dimension), dtype=np.float32)

    def _build_index(self) -> faiss.Index:
        d = self._dimension
        if self._config.index_type == "flat":
            index = faiss.IndexFlatIP(d)
        elif self._config.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(
                quantizer, d, self._config.n_lists, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self._config.n_probe
        elif self._config.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index type: {self._config.index_type}")
        logger.info("FAISS index built: type=%s, dim=%s", self._config.index_type, d)
        return index

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with their pre-computed embeddings."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk/embedding count mismatch")
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                "Vector dimension mismatch: "
                f"got {embeddings.shape[1]}, expected {self._dimension}. "
                "Rebuild the index or align the embedding model dimension."
            )

        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            logger.info("Training IVF index...")
            self._index.train(embeddings)
        self._index.add(embeddings.astype(np.float32))
        # persist embeddings in-memory for MMR and other operations
        if self._embeddings is None or self._embeddings.size == 0:
            self._embeddings = embeddings.astype(np.float32).copy()
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype(np.float32)])

        self._chunks.extend(chunks)
        logger.info("Added %s chunks. Total: %s", len(chunks), self._index.ntotal)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        if query_embedding.shape[1] != self._dimension:
            raise ValueError(
                "Query dimension mismatch: "
                f"got {query_embedding.shape[1]}, expected {self._dimension}."
            )

        scores, indices = self._index.search(query_embedding, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            clamped = float(np.clip(score, 0.0, 1.0))
            if clamped < threshold:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], similarity_score=clamped))
        return results

    def mmr_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        threshold: float = 0.0,
    ) -> list[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        if query_embedding.shape[1] != self._dimension:
            raise ValueError(
                "Query dimension mismatch: "
                f"got {query_embedding.shape[1]}, expected {self._dimension}."
            )

        actual_fetch = min(fetch_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, actual_fetch)

        candidates: list[_Candidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidates.append(_Candidate(
                chunk=self._chunks[idx],
                raw_score=float(score),
                faiss_index=int(idx),
            ))

        if not candidates:
            return []

        # Prefer stored embeddings to avoid using faiss.reconstruct(), which fails
        # on some index types (e.g. PQ). If embeddings are unavailable, attempt
        # to reconstruct per-candidate and gracefully fallback to top-k.
        if self._embeddings is not None and self._embeddings.shape[0] >= len(self._chunks):
            # Use stored embeddings by mapping faiss indices to rows
            cand_embs = np.vstack([self._embeddings[c.faiss_index] for c in candidates])
        else:
            tried = []
            failed = False
            for c in candidates:
                try:
                    tried.append(self._reconstruct(c.faiss_index))
                except NotImplementedError:
                    failed = True
                    break
            if failed:
                # graceful fallback: return top-k by raw score
                logger.warning("MMR fallback: stored embeddings unavailable and reconstruct unsupported; returning top-k by raw score")
                results = []
                for cand in candidates[:top_k]:
                    clamped_score = float(np.clip(cand.raw_score, 0.0, 1.0))
                    if clamped_score < threshold:
                        continue
                    results.append(RetrievedChunk(chunk=cand.chunk, similarity_score=clamped_score))
                return results
            cand_embs = np.array(tried)
        query_sims = (cand_embs @ query_embedding.T).flatten()

        selected_idxs: list[int] = []
        remaining = list(range(len(candidates)))

        while len(selected_idxs) < top_k and remaining:
            best_idx = -1
            best_score = -float("inf")

            for i in remaining:
                relevance = query_sims[i]
                if selected_idxs:
                    sel_embs = cand_embs[selected_idxs]
                    max_sim_to_selected = float((cand_embs[i] @ sel_embs.T).max())
                else:
                    max_sim_to_selected = 0.0

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected_idxs.append(best_idx)
            remaining.remove(best_idx)

        results = []
        for i in selected_idxs:
            clamped_score = float(np.clip(candidates[i].raw_score, 0.0, 1.0))
            if clamped_score < threshold:
                continue
            results.append(RetrievedChunk(chunk=candidates[i].chunk, similarity_score=clamped_score))

        return results

    def save(self, name: str = "default") -> None:
        directory = Path(self._config.persist_dir) / name
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))
        payload = {
            "dimension": self._dimension,
            "chunks": [c.model_dump() for c in self._chunks],
        }
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)
        # Save embeddings in a binary npy file for efficient load
        try:
            if self._embeddings is not None and self._embeddings.size:
                np.save(str(directory / "embeddings.npy"), self._embeddings)
        except Exception:
            logger.exception("Failed to save embeddings.npy; continuing")
        logger.info("Vector store saved to %s", directory)

    def load(self, name: str = "default") -> None:
        directory = Path(self._config.persist_dir) / name
        self._index = faiss.read_index(str(directory / "index.faiss"))
        if self._index.d != self._dimension:
            raise ValueError(
                "Persisted index dimension mismatch: "
                f"index has {self._index.d}, but VectorStore expects {self._dimension}. "
                "Rebuild the persisted index with the current embedding model."
            )

        with open(directory / "chunks.json", encoding="utf-8") as f:
            payload = json.load(f)
            chunks = payload["chunks"] if isinstance(payload, dict) else payload
            persisted_dim = payload.get("dimension") if isinstance(payload, dict) else self._dimension
            if persisted_dim != self._dimension:
                raise ValueError(
                    "Persisted chunk metadata dimension mismatch: "
                    f"got {persisted_dim}, expected {self._dimension}."
                )
            self._chunks = [Chunk(**c) for c in chunks]
        # Try to load stored embeddings.npy
        emb_path = directory / "embeddings.npy"
        if emb_path.exists():
            try:
                self._embeddings = np.load(str(emb_path))
                if self._embeddings.shape[1] != self._dimension:
                    raise ValueError("Persisted embeddings dimension mismatch")
            except Exception:
                logger.exception("Failed to load embeddings.npy; will attempt reconstruct fallback")
                self._embeddings = None
        else:
            # Attempt reconstructing embeddings into memory for backward compatibility
            try:
                embs = []
                for i in range(self._index.ntotal):
                    embs.append(self._reconstruct(i))
                if embs:
                    self._embeddings = np.vstack(embs).astype(np.float32)
                else:
                    self._embeddings = np.zeros((0, self._dimension), dtype=np.float32)
            except NotImplementedError:
                logger.warning("Index does not support reconstruct(); embeddings not loaded. MMR will fallback to top-k when needed.")
                self._embeddings = None

        logger.info("Loaded %s vectors from %s (embeddings_loaded=%s)", self._index.ntotal, directory, self._embeddings is not None)

    def _reconstruct(self, idx: int) -> np.ndarray:
        try:
            return self._index.reconstruct(idx)
        except RuntimeError:
            raise NotImplementedError(
                "Reconstruction not supported for this index type. Use 'flat' or 'hnsw' for MMR support."
            )

    @property
    def size(self) -> int:
        return self._index.ntotal
