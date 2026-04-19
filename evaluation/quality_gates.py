"""Quality regression gates for Phase 2 evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAITHFULNESS_MIN = 0.90
ANSWER_RELEVANCY_MIN = 0.95


def load_report(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metrics" not in payload or not isinstance(payload["metrics"], dict):
        raise ValueError("Invalid report: missing metrics object")
    return payload


def _get_metrics(report: dict) -> dict:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Invalid report: missing metric keys: metrics")
    return metrics


def evaluate_quality_gates(report: dict) -> tuple[bool, list[str]]:
    metrics = _get_metrics(report)
    missing = [key for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall") if key not in metrics]
    if missing:
        raise ValueError(f"Invalid report: missing metric keys: {', '.join(missing)}")

    messages: list[str] = []
    faithfulness = float(metrics["faithfulness"])
    answer_relevancy = float(metrics["answer_relevancy"])
    context_precision = float(metrics["context_precision"])
    context_recall = float(metrics["context_recall"])

    passed = True

    if faithfulness < FAITHFULNESS_MIN:
        passed = False
        messages.append(
            f"FAIL faithfulness={faithfulness:.4f} < {FAITHFULNESS_MIN:.2f}"
        )
    else:
        messages.append(
            f"PASS faithfulness={faithfulness:.4f} >= {FAITHFULNESS_MIN:.2f}"
        )

    if answer_relevancy < ANSWER_RELEVANCY_MIN:
        passed = False
        messages.append(
            f"FAIL answer_relevancy={answer_relevancy:.4f} < {ANSWER_RELEVANCY_MIN:.2f}"
        )
    else:
        messages.append(
            f"PASS answer_relevancy={answer_relevancy:.4f} >= {ANSWER_RELEVANCY_MIN:.2f}"
        )

    messages.append(f"INFO context_precision={context_precision:.4f} (non-blocking)")
    messages.append(f"INFO context_recall={context_recall:.4f} (non-blocking)")
    return passed, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 quality regression gates")
    parser.add_argument("--report", required=True, help="Path to evaluation report JSON")
    args = parser.parse_args()

    report = load_report(args.report)
    passed, messages = evaluate_quality_gates(report)
    for msg in messages:
        print(msg)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
