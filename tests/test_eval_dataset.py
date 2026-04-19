import json

import pytest

from evaluation.dataset import load_evaluation_dataset


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _sample_row(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "question": "What is RAG?",
        "ground_truth_answer": "Retrieval Augmented Generation",
        "expected_context_ids": ["ctx-1", "ctx-2"],
        "source_documents": ["doc-a", "doc-b"],
        "synthetic": False,
    }


def test_load_valid_rows(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    _write_jsonl(dataset_path, [_sample_row("a"), _sample_row("b")])

    rows = load_evaluation_dataset(dataset_path)

    assert len(rows) == 2
    assert rows[0].id == "a"
    assert rows[1].id == "b"


def test_missing_required_key_fails(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    row = _sample_row("a")
    del row["question"]
    _write_jsonl(dataset_path, [row])

    with pytest.raises(ValueError, match="missing required keys"):
        load_evaluation_dataset(dataset_path)


def test_malformed_type_fails(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    row = _sample_row("a")
    row["expected_context_ids"] = "ctx-1"
    _write_jsonl(dataset_path, [row])

    with pytest.raises(ValueError, match="Invalid evaluation row"):
        load_evaluation_dataset(dataset_path)


def test_loader_preserves_file_order(tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        dataset_path,
        [_sample_row("third"), _sample_row("first"), _sample_row("second")],
    )

    rows = load_evaluation_dataset(dataset_path)

    assert [r.id for r in rows] == ["third", "first", "second"]


def test_phase2_seed_dataset_exists_and_is_valid():
    dataset_path = "evaluation/datasets/phase2_eval.jsonl"
    rows = load_evaluation_dataset(dataset_path)

    assert len(rows) >= 1
    assert rows[0].id
