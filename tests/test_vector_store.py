import numpy as np
import pytest
import tempfile

from config import VectorStoreConfig
from vector_store import VectorStore
from models import Chunk


@pytest.fixture
def store():
    config = VectorStoreConfig(persist_dir=tempfile.mkdtemp())
    return VectorStore(config, dimension=1024)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(document_id="d1", content="Hello world", chunk_index=0),
        Chunk(document_id="d1", content="Goodbye world", chunk_index=1),
        Chunk(document_id="d2", content="Foo bar baz", chunk_index=0),
    ]


@pytest.fixture
def deterministic_embeddings():
    embs = np.array([
        [1.0, 0.5, 0.3, 0.1] + [0.0] * 1020,
        [0.9, 0.6, 0.2, 0.2] + [0.0] * 1020,
        [0.3, 0.1, 0.9, 0.5] + [0.0] * 1020,
    ], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def test_add_and_search(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)
    assert store.size == 3

    results = store.search(deterministic_embeddings[0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.content == "Hello world"


def test_search_with_threshold(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)

    results = store.search(deterministic_embeddings[0], top_k=10, threshold=0.95)
    assert len(results) >= 1
    assert results[0].chunk.content == "Hello world"


def test_persistence(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)
    store.save("test_persist")

    new_store = VectorStore(store._config, dimension=1024)
    new_store.load("test_persist")
    assert new_store.size == 3

    results = new_store.search(deterministic_embeddings[0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.content == "Hello world"


def test_empty_search(store):
    query = np.random.randn(1024).astype(np.float32)
    results = store.search(query, top_k=5)
    assert results == []


def test_mmr_search(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)

    query = np.array([1.0, 0.4, 0.3, 0.1] + [0.0] * 1020, dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(query, top_k=2, fetch_k=3, lambda_mult=0.7)

    assert len(results) == 2
    assert results[0].chunk.content == "Hello world"


def test_mmr_search_top_k_larger_than_store(store, sample_chunks, deterministic_embeddings):
    store.add(sample_chunks, deterministic_embeddings)

    query = np.array([1.0, 0.4, 0.3, 0.1] + [0.0] * 1020, dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(query, top_k=10, fetch_k=20, lambda_mult=0.7)
    assert len(results) == 3


def test_mmr_promotes_diversity(store):
    near_dup_chunks = [
        Chunk(document_id="d1", content="Machine learning is great", chunk_index=0),
        Chunk(document_id="d1", content="Machine learning is wonderful", chunk_index=1),
        Chunk(document_id="d2", content="Quantum physics is fascinating", chunk_index=0),
    ]

    embs = np.array([
        [0.9, 0.1, 0.0, 0.0] + [0.0] * 1020,
        [0.88, 0.12, 0.0, 0.0] + [0.0] * 1020,
        [0.5, 0.0, 0.5, 0.0] + [0.0] * 1020,
    ], dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    store.add(near_dup_chunks, embs)

    query = np.array([0.92, 0.08, 0.02, 0.0] + [0.0] * 1020, dtype=np.float32)
    query /= np.linalg.norm(query)

    results = store.mmr_search(query, top_k=2, fetch_k=3, lambda_mult=0.5)

    assert len(results) == 2
    contents = [r.chunk.content for r in results]
    assert "Machine learning is great" in contents
    assert "Quantum physics is fascinating" in contents
    assert "Machine learning is wonderful" not in contents


def test_add_rejects_wrong_embedding_width(store, sample_chunks):
    bad_embeddings = np.zeros((3, 384), dtype=np.float32)

    with pytest.raises(ValueError) as exc:
        store.add(sample_chunks, bad_embeddings)

    assert "Vector dimension mismatch" in str(exc.value)


def test_load_rejects_incompatible_dimension(tmp_path):
    config = VectorStoreConfig(persist_dir=str(tmp_path))
    store = VectorStore(config, dimension=1024)
    store.add([
        Chunk(document_id="d1", content="Hello world", chunk_index=0),
    ], np.ones((1, 1024), dtype=np.float32))
    store.save("persisted")

    incompatible = VectorStore(config, dimension=384)
    with pytest.raises(ValueError) as exc:
        incompatible.load("persisted")

    assert "Persisted index dimension mismatch" in str(exc.value)
