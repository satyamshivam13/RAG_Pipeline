"""Tests for the Generator (uses mocked LLM)."""
import pytest
from unittest.mock import MagicMock

from config import GeneratorConfig
from generator import Generator
from models import RetrievedChunk, Chunk


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat.return_value = "Based on the context, the answer is 42."
    llm.count_tokens.side_effect = lambda text: max(1, len(text) // 4)
    return llm


@pytest.fixture
def gen(mock_llm):
    return Generator(GeneratorConfig(), mock_llm)


def test_generate_with_context(gen, mock_llm):
    """Generator should call LLM and return the answer when context exists."""
    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d1", content="The answer is 42."),
            similarity_score=0.95,
        )
    ]

    output = gen.generate("What is the answer?", chunks)

    assert "42" in output.answer
    assert output.query == "What is the answer?"
    assert len(output.context_used) == 1
    assert output.context_truncated is False
    mock_llm.chat.assert_called_once()


def test_generate_no_context(gen):
    """Without context, generator should decline to answer (not hallucinate)."""
    output = gen.generate("Random question?", [])

    assert "don't have enough" in output.answer.lower()
    assert output.context_used == []


def test_generate_multiple_chunks(gen, mock_llm):
    """Generator should pass all context chunks to the LLM when under budget."""
    chunks = [
        RetrievedChunk(
            chunk=Chunk(document_id="d1", content="Fact A."),
            similarity_score=0.9,
        ),
        RetrievedChunk(
            chunk=Chunk(document_id="d2", content="Fact B."),
            similarity_score=0.8,
        ),
    ]

    output = gen.generate("Tell me facts", chunks)

    assert len(output.context_used) == 2
    call_args = mock_llm.chat.call_args
    user_msg = call_args.kwargs.get("messages", call_args[0][0] if call_args[0] else [])
    prompt_text = str(user_msg)
    assert "Fact A" in prompt_text
    assert "Fact B" in prompt_text


def test_truncates_least_relevant_chunks_when_over_budget(mock_llm):
    cfg = GeneratorConfig(max_context_tokens=20)
    generator = Generator(cfg, mock_llm)

    chunks = [
        RetrievedChunk(chunk=Chunk(document_id="d1", content="A" * 120), similarity_score=0.95),
        RetrievedChunk(chunk=Chunk(document_id="d2", content="B" * 120), similarity_score=0.55),
        RetrievedChunk(chunk=Chunk(document_id="d3", content="C" * 120), similarity_score=0.25),
    ]

    output = generator.generate("q", chunks)

    assert output.context_truncated is True
    assert len(output.context_used) == 1
    assert output.context_used[0].similarity_score == 0.95
    assert any("Context truncated" in w for w in output.warnings)
