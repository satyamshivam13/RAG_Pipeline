import json
from pathlib import Path

from evaluation.run_eval import METRIC_KEYS, run_evaluation
from models import Chunk, GeneratorOutput, PipelineResult, RetrievedChunk, EvaluatorOutput


def _write_dataset(path):
    row = {
        "id": "row-1",
        "question": "What does RAG stand for?",
        "ground_truth_answer": "Retrieval Augmented Generation",
        "expected_context_ids": ["ctx-1"],
        "source_documents": ["doc-1"],
        "synthetic": False,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


class _StubPipeline:
    def __init__(self, *_args, **_kwargs):
        pass

    def query(self, question: str) -> PipelineResult:
        chunk = Chunk(id="ctx-1", document_id="doc-1", content="retrieval augmented generation", source="doc-1")
        retrieved = [RetrievedChunk(chunk=chunk, similarity_score=0.9)]
        gen = GeneratorOutput(
            answer="Retrieval Augmented Generation.",
            query=question,
            context_used=retrieved,
            model="stub",
            processing_time_ms=1.0,
        )
        eval_output = EvaluatorOutput(
            overall_consistency_score=0.95,
            is_reliable=True,
            claims=[],
            summary="ok",
            processing_time_ms=1.0,
        )
        return PipelineResult(
            query=question,
            answer=gen.answer,
            is_reliable=True,
            consistency_score=0.95,
            retrieval=retrieved,
            guardrail=None,
            generation=gen,
            evaluation=eval_output,
            total_time_ms=5.0,
        )

    def close(self):
        return None


def test_run_eval_writes_required_report_keys(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset)

    monkeypatch.setattr("evaluation.run_eval.RAGPipeline", _StubPipeline)

    report = run_evaluation(dataset, report_path, now_provider=lambda: "2026-04-19T00:00:00Z")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["phase"] == "02-measure-observe"
    assert report["dataset_size"] == 1
    assert set(report["metrics"].keys()) == set(METRIC_KEYS)
    assert set(loaded["metrics"].keys()) == set(METRIC_KEYS)


def test_run_eval_output_shape_is_stable(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset)

    monkeypatch.setattr("evaluation.run_eval.RAGPipeline", _StubPipeline)

    first = run_evaluation(dataset, report_path, now_provider=lambda: "fixed")
    second = run_evaluation(dataset, report_path, now_provider=lambda: "fixed")

    assert first.keys() == second.keys()
    assert first["metrics"].keys() == second["metrics"].keys()


def test_eval_cli_smoke_with_stubs(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    report_path = tmp_path / "report.json"
    _write_dataset(dataset)

    monkeypatch.setattr("evaluation.run_eval.RAGPipeline", _StubPipeline)

    report = run_evaluation(
        dataset,
        report_path,
        now_provider=lambda: "2026-04-19T00:00:00Z",
    )

    assert Path(report_path).exists()
    assert set(report["metrics"].keys()) == set(METRIC_KEYS)
