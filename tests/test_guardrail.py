"""Tests for the Guardrail Agent (uses mocked LLM)."""
import pytest
from unittest.mock import MagicMock

from config import GuardrailConfig
from guardrail_agent import GuardrailAgent
from models import RetrievedChunk, Chunk, RelevanceVerdict


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def agent(mock_llm):
    return GuardrailAgent(GuardrailConfig(), mock_llm)


@pytest.fixture
def sample_chunks():
    c1 = Chunk(id="c1", document_id="d1", content="Relevant content about AI")
    c2 = Chunk(id="c2", document_id="d1", content="Irrelevant content about cooking")
    return [
        RetrievedChunk(chunk=c1, similarity_score=0.9),
        RetrievedChunk(chunk=c2, similarity_score=0.4),
    ]


def test_guardrail_filters_irrelevant(agent, mock_llm, sample_chunks):
    """Chunks below the relevance threshold should be removed."""
    mock_llm.chat_json.return_value = {
        "evaluations": [
            {"chunk_id": "c1", "relevance_score": 0.9, "verdict": "relevant",
             "reasoning": "Directly about AI"},
            {"chunk_id": "c2", "relevance_score": 0.2, "verdict": "irrelevant",
             "reasoning": "About cooking, not AI"},
        ],
        "safety_flags": [],
    }

    output = agent.evaluate("What is AI?", sample_chunks)

    assert len(output.filtered_chunks) == 1
    assert output.filtered_chunks[0].chunk.id == "c1"
    assert len(output.removed_chunks) == 1


def test_guardrail_empty_input(agent):
    output = agent.evaluate("test query", [])
    assert output.original_count == 0
    assert output.filtered_chunks == []


def test_guardrail_safety_flags(agent, mock_llm, sample_chunks):
    mock_llm.chat_json.return_value = {
        "evaluations": [
            {"chunk_id": "c1", "relevance_score": 0.9, "verdict": "relevant",
             "reasoning": "Relevant"},
            {"chunk_id": "c2", "relevance_score": 0.8, "verdict": "relevant",
             "reasoning": "Relevant"},
        ],
        "safety_flags": ["Contains email address (PII)"],
    }

    output = agent.evaluate("test", sample_chunks)
    assert len(output.safety_flags) == 1
    assert "PII" in output.safety_flags[0]