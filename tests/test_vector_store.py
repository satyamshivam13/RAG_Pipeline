"""Tests for the vector store."""
import numpy as np
import pytest
import tempfile

from config import VectorStoreConfig
from vector_store import VectorStore
from models import Chunk


@pytest.fixture
def store():
    config = VectorStoreConfig(persist_dir=tempfile.mkdtemp())
    return VectorStore(config, dimension=4)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(document_id="d1", content="Hello world", chunk_index=0),
        Chunk(document_id="d1", content="Goodbye world", chunk_index=1),
        Chunk(document_id="d2", content="Foo bar baz", chunk_index=0),
    ]


@pytest.fixture
def deterministic_embeddings():
    """
    Hand-crafted normalized embeddings with KNOWN similarity relationships.
    All components positive → all pairwise cosine similarities > 0.
    """
    embs = np.array([
        [1.0, 0.5, 0.3, 0.1],   # chunk 0
        [0.9, 0.6, 0.2, 0.2],   # chunk 1 — similar to chunk 0
        [0.3, 0.1, 0.9, 0.5],   # chunk 2 — different direction
    ], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def test_add_and_search(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)
    assert store.size == 3

    results = store.search(deterministic_embeddings[0], top_k=2)
    assert len(results) == 2
    # First result is exact match → score ≈ 1.0
    assert results[0].chunk.content == "Hello world"
    assert results[0].similarity_score == pytest.approx(1.0, abs=0.01)


def test_search_with_threshold(store, sample_chunks, deterministic_embeddings):
    """Chunks below the similarity threshold should be excluded."""
    store.add(sample_chunks, deterministic_embeddings)

    results = store.search(deterministic_embeddings[0], top_k=10, threshold=0.95)
    assert len(results) >= 1
    assert results[0].chunk.content == "Hello world"


def test_persistence(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)
    store.save("test_persist")

    new_store = VectorStore(store._config, dimension=4)
    new_store.load("test_persist")
    assert new_store.size == 3

    results = new_store.search(deterministic_embeddings[0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.content == "Hello world"


def test_empty_search(store):
    """Searching an empty store should return an empty list."""
    query = np.random.randn(4).astype(np.float32)
    results = store.search(query, top_k=5)
    assert results == []


def test_mmr_search(store, sample_chunks, deterministic_embeddings):
    """MMR should return the requested number of diverse results."""
    store.add(sample_chunks, deterministic_embeddings)

    # ═══════════════════════════════════════════════════════════
    # KEY FIX: query must NOT be identical to a stored embedding.
    #
    # WHY: If query == stored_doc[0], then after selecting doc[0]:
    #   sim(candidate, query) == sim(candidate, selected[0])
    #   MMR = λ*rel - (1-λ)*rel = (2λ-1)*rel
    #   → Diversity term cancels out completely!
    # ═══════════════════════════════════════════════════════════
    query = np.array([1.0, 0.4, 0.3, 0.1], dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(
        query, top_k=2, fetch_k=3, lambda_mult=0.7,
    )

    assert len(results) == 2
    # First result should be most similar to query
    assert results[0].chunk.content == "Hello world"


def test_mmr_search_top_k_larger_than_store(store, sample_chunks, deterministic_embeddings):
    """Requesting more results than exist should return all available."""
    store.add(sample_chunks, deterministic_embeddings)

    query = np.array([1.0, 0.4, 0.3, 0.1], dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(
        query, top_k=10, fetch_k=20, lambda_mult=0.7,
    )
    assert len(results) == 3


def test_mmr_promotes_diversity(store):
    """
    Given two near-identical chunks and one different chunk,
    MMR with λ=0.5 should prefer the different one for the 2nd slot.

    Math proof for these exact vectors:
    ─────────────────────────────────────────────────────
    After selecting chunk 0 (most relevant):

      chunk 1 MMR = 0.5 × 0.999 − 0.5 × 0.9997 = −0.0004
      chunk 2 MMR = 0.5 × 0.720 − 0.5 × 0.7027 = +0.0086  ← WINS

    Chunk 2 wins because its diversity bonus (low similarity
    to the already-selected chunk 0) outweighs its lower
    relevance to the query.
    ─────────────────────────────────────────────────────
    """
    near_dup_chunks = [
        Chunk(document_id="d1", content="Machine learning is great", chunk_index=0),
        Chunk(document_id="d1", content="Machine learning is wonderful", chunk_index=1),
        Chunk(document_id="d2", content="Quantum physics is fascinating", chunk_index=0),
    ]

    # Chunk 0 and 1: nearly identical direction (cosine ≈ 0.9997)
    # Chunk 2: different direction but still has positive relevance
    embs = np.array([
        [0.9, 0.1, 0.0, 0.0],    # chunk 0 — "ML is great"
        [0.88, 0.12, 0.0, 0.0],  # chunk 1 — near-duplicate
        [0.5, 0.0, 0.5, 0.0],    # chunk 2 — different topic
    ], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    store.add(near_dup_chunks, embs)

    # ═══════════════════════════════════════════════════════════
    # Query is CLOSE TO but NOT IDENTICAL TO chunk 0.
    # This ensures sim(candidate, query) ≠ sim(candidate, selected)
    # so the diversity term actually has an effect.
    # ═══════════════════════════════════════════════════════════
    query = np.array([0.92, 0.08, 0.02, 0.0], dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(
        query,
        top_k=2,
        fetch_k=3,
        lambda_mult=0.5,  # Equal weight: relevance vs diversity
    )

    assert len(results) == 2
    contents = [r.chunk.content for r in results]

    # Chunk 0 selected first (highest relevance)
    assert "Machine learning is great" in contents

    # Chunk 2 selected second (diversity wins over near-duplicate)
    assert "Quantum physics is fascinating" in contents

    # Near-duplicate should be excluded
    assert "Machine learning is wonderful" not in contents