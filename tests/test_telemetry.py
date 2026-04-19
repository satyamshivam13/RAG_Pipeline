import pytest

from config import TelemetryConfig
from telemetry import (
    get_correlation_id,
    get_or_create_correlation_id,
    resolve_exporter_config,
    reset_correlation_id as clear_correlation_id,
    set_correlation_id,
)
from dataclasses import replace

from config import PipelineConfig, RuntimeConfig
from main import RAGPipeline
from models import Chunk, RetrievedChunk, GeneratorOutput, EvaluatorOutput


@pytest.fixture(autouse=True)
def reset_correlation_id():
    yield
    clear_correlation_id()


class _StubEmbeddingModel:
    def __init__(self, _cfg):
        self.dimension = 4

    def embed(self, texts):
        return []

    def embed_query(self, query):
        return [0.1, 0.2, 0.3, 0.4]


class _StubVectorStore:
    def __init__(self, _cfg, dimension):
        self._dimension = dimension

    def add(self, chunks, embeddings):
        return None

    def save(self, name):
        return None

    def load(self, name):
        return None


class _StubLoader:
    def __init__(self, _cfg):
        pass

    def chunk_documents(self, docs):
        return []


class _StubRetriever:
    def __init__(self, _cfg, _emb, _store):
        pass

    def retrieve(self, query):
        c1 = Chunk(id="ctx-a", document_id="d1", content="ctx 1", source="s1")
        c2 = Chunk(id="ctx-b", document_id="d1", content="ctx 2", source="s1")
        return [
            RetrievedChunk(chunk=c1, similarity_score=0.9),
            RetrievedChunk(chunk=c2, similarity_score=0.4),
        ]


class _StubGenerator:
    def __init__(self, cfg, _llm):
        self._model = cfg.model

    def generate(self, query, context_chunks):
        return GeneratorOutput(
            answer="stub answer",
            query=query,
            context_used=context_chunks,
            model=self._model,
            processing_time_ms=1.0,
        )


class _StubGuardrail:
    def __init__(self, _cfg, _llm):
        pass

    def evaluate(self, query, chunks):
        return None


class _StubEvaluator:
    def __init__(self, _cfg, _llm):
        pass

    def evaluate(self, answer, context_chunks, query=""):
        return EvaluatorOutput(
            overall_consistency_score=0.9,
            is_reliable=True,
            claims=[],
            summary="ok",
            processing_time_ms=10.0,
        )


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr("main.LLMClient", lambda cfg: object())
    monkeypatch.setattr("main.EmbeddingModel", _StubEmbeddingModel)
    monkeypatch.setattr("main.VectorStore", _StubVectorStore)
    monkeypatch.setattr("main.DocumentLoader", _StubLoader)
    monkeypatch.setattr("main.Retriever", _StubRetriever)
    monkeypatch.setattr("main.Generator", _StubGenerator)
    monkeypatch.setattr("main.GuardrailAgent", _StubGuardrail)
    monkeypatch.setattr("main.EvaluatorAgent", _StubEvaluator)


def test_get_or_create_correlation_id_is_stable_within_context():
    set_correlation_id("test-correlation")
    first = get_or_create_correlation_id()
    second = get_or_create_correlation_id()

    assert first == "test-correlation"
    assert second == "test-correlation"
    assert get_correlation_id() == "test-correlation"


def test_exporter_defaults_to_console_and_accepts_otlp():
    default_cfg = TelemetryConfig(
        telemetry_enabled=True,
        telemetry_service_name="rag",
        telemetry_exporter="",
        telemetry_otlp_endpoint=None,
    )
    default_selection = resolve_exporter_config(default_cfg)
    assert default_selection.exporter == "console"
    assert default_selection.endpoint is None

    otlp_cfg = TelemetryConfig(
        telemetry_enabled=True,
        telemetry_service_name="rag",
        telemetry_exporter="otlp",
        telemetry_otlp_endpoint="http://collector:4318/v1/traces",
    )
    otlp_selection = resolve_exporter_config(otlp_cfg)
    assert otlp_selection.exporter == "otlp"
    assert otlp_selection.endpoint == "http://collector:4318/v1/traces"


def test_sync_and_deferred_modes_keep_correlation_id(monkeypatch):
    _patch_pipeline(monkeypatch)
    set_correlation_id("corr-sync")
    sync_cfg = replace(PipelineConfig(), runtime=RuntimeConfig(evaluator_mode="sync"))

    pipe_sync = RAGPipeline(sync_cfg)
    try:
        _ = pipe_sync.query("what?")
        assert get_correlation_id() == "corr-sync"
    finally:
        pipe_sync.close()

    _patch_pipeline(monkeypatch)
    set_correlation_id("corr-deferred")
    deferred_cfg = replace(PipelineConfig(), runtime=RuntimeConfig(evaluator_mode="deferred"))

    pipe_deferred = RAGPipeline(deferred_cfg)
    try:
        _ = pipe_deferred.query("what?")
        assert get_correlation_id() == "corr-deferred"
    finally:
        pipe_deferred.close()
