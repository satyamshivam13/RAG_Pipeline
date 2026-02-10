"""Tests for the Evaluator Agent (uses mocked LLM)."""
import pytest
from unittest.mock import MagicMock

from config import EvaluatorConfig
from evaluator_agent import EvaluatorAgent
from models import RetrievedChunk, Chunk, ClaimVerdict


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def evaluator(mock_llm):
    return EvaluatorAgent(EvaluatorConfig(), mock_llm)


def test_evaluator_high_consistency(evaluator, mock_llm):
    """Fully supported claims should produce a high consistency score."""
    mock_llm.chat_json.return_value = {
        "claims": [
            {
                "claim": "The sky is blue",
                "verdict": "supported",
                "supporting_evidence": "Context says sky is blue",
                "reasoning": "Directly stated",
            },
        ],
        "overall_consistency_score": 1.0,
        "summary": "Fully consistent.",
    }

    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d1", content="The sky is blue."),
            similarity_score=0.9,
        ),
    ]

    output = evaluator.evaluate("What color is the sky?", chunks, query="What color?")

    assert output.overall_consistency_score == 1.0
    assert output.is_reliable is True
    assert len(output.claims) == 1
    assert output.claims[0].verdict == ClaimVerdict.SUPPORTED


def test_evaluator_low_consistency(evaluator, mock_llm):
    """Contradicted claims should produce a low score and unreliable flag."""
    mock_llm.chat_json.return_value = {
        "claims": [
            {
                "claim": "The sky is green",
                "verdict": "contradicted",
                "supporting_evidence": "Context says sky is blue, not green",
                "reasoning": "Directly contradicts the source",
            },
        ],
        "overall_consistency_score": 0.1,
        "summary": "Answer contradicts the source material.",
    }

    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d1", content="The sky is blue."),
            similarity_score=0.9,
        ),
    ]

    output = evaluator.evaluate("What color is the sky?", chunks, query="What color?")

    assert output.overall_consistency_score < 0.5
    assert output.is_reliable is False
    assert output.claims[0].verdict == ClaimVerdict.CONTRADICTED


def test_evaluator_empty_answer(evaluator):
    """Empty answer or no context should return score 0 and unreliable."""
    output = evaluator.evaluate("", [], query="test")
    assert output.overall_consistency_score == 0.0
    assert output.is_reliable is False
    assert output.claims == []


def test_evaluator_mixed_claims(evaluator, mock_llm):
    """Mix of supported and unsupported claims should produce a middling score."""
    mock_llm.chat_json.return_value = {
        "claims": [
            {
                "claim": "Python was created in 1991",
                "verdict": "supported",
                "supporting_evidence": "Context confirms 1991",
                "reasoning": "Exact match",
            },
            {
                "claim": "Python is the fastest language",
                "verdict": "not_supported",
                "supporting_evidence": "",
                "reasoning": "Context does not discuss performance",
            },
        ],
        "overall_consistency_score": 0.5,
        "summary": "One claim supported, one not mentioned in context.",
    }

    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d1", content="Python was created in 1991."),
            similarity_score=0.9,
        ),
    ]

    output = evaluator.evaluate("Tell me about Python", chunks)

    assert output.overall_consistency_score == 0.5
    assert len(output.claims) == 2