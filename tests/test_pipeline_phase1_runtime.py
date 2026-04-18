import time
from dataclasses import replace

from config import PipelineConfig, RuntimeConfig
from main import RAGPipeline
from models import Chunk, RetrievedChunk, GeneratorOutput, EvaluatorOutput, EvaluationStatus


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
        c1 = Chunk(id="c1", document_id="d1", content="ctx 1", source="s1")
        c2 = Chunk(id="c2", document_id="d1", content="ctx 2", source="s1")
        return [
            RetrievedChunk(chunk=c1, similarity_score=0.9),
            RetrievedChunk(chunk=c2, similarity_score=0.2),
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
    called = 0

    def __init__(self, _cfg, _llm):
        pass

    def evaluate(self, query, chunks):
        _StubGuardrail.called += 1
        raise AssertionError("guardrail should not run in default runtime mode")


class _SlowEvaluator:
    def __init__(self, _cfg, _llm):
        pass

    def evaluate(self, answer, context_chunks, query=""):
        time.sleep(0.15)
        return EvaluatorOutput(
            overall_consistency_score=0.9,
            is_reliable=True,
            claims=[],
            summary="ok",
            processing_time_ms=150.0,
        )


def _patch_pipeline(monkeypatch, evaluator_cls):
    monkeypatch.setattr("main.LLMClient", lambda cfg: object())
    monkeypatch.setattr("main.EmbeddingModel", _StubEmbeddingModel)
    monkeypatch.setattr("main.VectorStore", _StubVectorStore)
    monkeypatch.setattr("main.DocumentLoader", _StubLoader)
    monkeypatch.setattr("main.Retriever", _StubRetriever)
    monkeypatch.setattr("main.Generator", _StubGenerator)
    monkeypatch.setattr("main.GuardrailAgent", _StubGuardrail)
    monkeypatch.setattr("main.EvaluatorAgent", evaluator_cls)


def test_query_default_mode_skips_guardrail_and_defers_evaluation(monkeypatch):
    _StubGuardrail.called = 0
    _patch_pipeline(monkeypatch, _SlowEvaluator)

    cfg = PipelineConfig()
    pipe = RAGPipeline(cfg)
    try:
        started = time.perf_counter()
        result = pipe.query("what?")
        elapsed = time.perf_counter() - started

        assert _StubGuardrail.called == 0
        assert result.evaluation_status == EvaluationStatus.PENDING
        assert result.evaluation_deferred is True
        assert result.consistency_score == 0.0
        assert elapsed < 0.12
    finally:
        pipe.close()


def test_query_sync_mode_waits_for_evaluator(monkeypatch):
    _patch_pipeline(monkeypatch, _SlowEvaluator)

    cfg = replace(PipelineConfig(), runtime=RuntimeConfig(evaluator_mode="sync"))
    pipe = RAGPipeline(cfg)
    try:
        started = time.perf_counter()
        result = pipe.query("what?")
        elapsed = time.perf_counter() - started

        assert result.evaluation_status == EvaluationStatus.COMPLETED
        assert result.evaluation_deferred is False
        assert result.consistency_score > 0.0
        assert elapsed >= 0.14
    finally:
        pipe.close()


def test_evaluator_failure_never_breaks_response_path(monkeypatch):
    class _FailEvaluator:
        def __init__(self, _cfg, _llm):
            pass

        def evaluate(self, answer, context_chunks, query=""):
            raise RuntimeError("boom")

    _patch_pipeline(monkeypatch, _FailEvaluator)

    cfg = replace(PipelineConfig(), runtime=RuntimeConfig(evaluator_mode="sync"))
    pipe = RAGPipeline(cfg)
    try:
        result = pipe.query("what?")

        assert result.answer == "stub answer"
        assert result.evaluation_status == EvaluationStatus.FAILED
        assert result.evaluation_error is not None
    finally:
        pipe.close()
