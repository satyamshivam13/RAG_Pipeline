import pytest

from evaluation.quality_gates import evaluate_quality_gates


def _report(faithfulness=0.95, answer_relevancy=0.96, context_precision=0.8, context_recall=0.82):
    return {
        "phase": "02-measure-observe",
        "metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        },
    }


def test_quality_gate_passes_when_thresholds_met():
    passed, messages = evaluate_quality_gates(_report())
    assert passed is True
    assert any("PASS faithfulness" in m for m in messages)
    assert any("PASS answer_relevancy" in m for m in messages)


def test_quality_gate_fails_when_faithfulness_below_threshold():
    passed, messages = evaluate_quality_gates(_report(faithfulness=0.89))
    assert passed is False
    assert any("FAIL faithfulness" in m for m in messages)


def test_quality_gate_fails_when_answer_relevancy_below_threshold():
    passed, messages = evaluate_quality_gates(_report(answer_relevancy=0.94))
    assert passed is False
    assert any("FAIL answer_relevancy" in m for m in messages)


def test_missing_metric_keys_raise_error():
    with pytest.raises(ValueError, match="missing metric keys"):
        evaluate_quality_gates({"metrics": {"faithfulness": 1.0}})


def test_missing_metrics_object_raise_error():
    with pytest.raises(ValueError, match="missing metric keys"):
        evaluate_quality_gates({"phase": "02-measure-observe"})
