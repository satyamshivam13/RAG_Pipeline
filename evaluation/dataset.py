"""Dataset contracts and loader for offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


class EvaluationSample(BaseModel):
    """Strictly validated evaluation dataset row."""

    id: str
    question: str
    ground_truth_answer: str
    expected_context_ids: list[str]
    source_documents: list[str]
    synthetic: bool


def _validate_required_keys(row: dict[str, Any], line_number: int) -> None:
    required_keys = {
        "id",
        "question",
        "ground_truth_answer",
        "expected_context_ids",
        "source_documents",
        "synthetic",
    }
    missing = sorted(required_keys - set(row.keys()))
    if missing:
        missing_txt = ", ".join(missing)
        raise ValueError(
            f"Invalid evaluation row at line {line_number}: missing required keys: {missing_txt}"
        )


def load_evaluation_dataset(path: str | Path) -> list[EvaluationSample]:
    """Load JSONL rows in file order with strict validation."""

    path = Path(path)
    samples: list[EvaluationSample] = []

    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {idx}: {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(f"Invalid evaluation row at line {idx}: expected JSON object")

            _validate_required_keys(row, idx)

            try:
                sample = EvaluationSample.model_validate(row)
            except ValidationError as exc:
                raise ValueError(f"Invalid evaluation row at line {idx}: {exc}") from exc

            samples.append(sample)

    return samples
