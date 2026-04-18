import numpy as np
import pytest

from config import EmbeddingConfig
from embeddings import EmbeddingModel


class _FakeSentenceTransformer:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        # Simulate bge-large dimension.
        return np.zeros((len(texts), 1024), dtype=np.float32)


class _MismatchSentenceTransformer:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        return np.zeros((len(texts), 384), dtype=np.float32)


def test_default_embedding_model_is_bge_large():
    cfg = EmbeddingConfig()
    assert cfg.model_name == "BAAI/bge-large-en-v1.5"
    assert cfg.dimension == 1024


def test_embedding_model_dimension_match(monkeypatch):
    monkeypatch.setattr("embeddings.SentenceTransformer", _FakeSentenceTransformer)
    cfg = EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5", dimension=1024)

    model = EmbeddingModel(cfg)

    assert model.dimension == 1024


def test_embedding_model_dimension_mismatch_fails_fast(monkeypatch):
    monkeypatch.setattr("embeddings.SentenceTransformer", _MismatchSentenceTransformer)
    cfg = EmbeddingConfig(model_name="all-MiniLM-L6-v2", dimension=1024)

    with pytest.raises(ValueError) as exc:
        EmbeddingModel(cfg)

    message = str(exc.value)
    assert "Embedding dimension mismatch" in message
    assert "outputs 384" in message
    assert "dimension is 1024" in message
