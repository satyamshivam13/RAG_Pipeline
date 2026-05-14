import time
import numpy as np
import pytest
import tempfile

from config import VectorStoreConfig
from vector_store import VectorStore
from models import Chunk


@pytest.mark.parametrize("index_type", ["flat", "ivf", "hnsw"])
def test_mmr_on_index_types(index_type):
    tmp = tempfile.mkdtemp()
    cfg = VectorStoreConfig(persist_dir=tmp, index_type=index_type, n_lists=2, n_probe=1)
    store = VectorStore(cfg, dimension=4)

    chunks = [Chunk(document_id=f"d{i}", content=f"doc {i}", chunk_index=i) for i in range(10)]

    rng = np.random.RandomState(42)
    embs = rng.randn(10, 4).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    # ensure add doesn't raise (IVF may require training internally)
    store.add(chunks, embs)

    q = rng.randn(4).astype(np.float32)
    q /= np.linalg.norm(q)

    results = store.mmr_search(q, top_k=3, fetch_k=5, lambda_mult=0.6)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_mmr_benchmark_small():
    cfg = VectorStoreConfig(persist_dir=tempfile.mkdtemp(), index_type="flat")
    store = VectorStore(cfg, dimension=64)

    rng = np.random.RandomState(0)
    n = 200
    chunks = [Chunk(document_id=str(i), content=str(i), chunk_index=i) for i in range(n)]
    embs = rng.randn(n, 64).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    store.add(chunks, embs)

    q = rng.randn(64).astype(np.float32)
    q /= np.linalg.norm(q)

    # baseline search
    t0 = time.time()
    sres = store.search(q, top_k=5)
    t_search = time.time() - t0

    t0 = time.time()
    mres = store.mmr_search(q, top_k=5, fetch_k=50, lambda_mult=0.6)
    t_mmr = time.time() - t0

    # MMR should complete and not be arbitrarily slower than search (10x is allowed)
    assert len(mres) <= 5
    assert t_mmr < max(0.5, 10 * max(t_search, 0.0001))
