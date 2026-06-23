"""
FastAPI API Integration Tests

Tests for all endpoints including health checks, ingest, and query operations.
These tests require OPENAI_API_KEY to be set for the app lifespan initialization.
"""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import json

from api import app, get_pipeline, limiter
from models import RetrievedChunk, Chunk, GeneratorOutput, EvaluatorOutput, EvaluationStatus, PipelineResult

# Skip all API tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


@pytest.fixture
def mock_pipeline():
    """Fixture to provide a mocked pipeline instance."""
    pipeline = MagicMock()

    # Mock ingest method
    pipeline.ingest = MagicMock(return_value=4)  # Returns number of chunks

    # Mock query method to return a realistic PipelineResult
    test_chunk = Chunk(
        document_id="test-1",
        content="RAG combines retrieval and generation for grounded answers.",
        source="test",
        chunk_index=0,
        metadata={"test": True},
    )
    retrieved = RetrievedChunk(chunk=test_chunk, similarity_score=0.92)

    gen_output = GeneratorOutput(
        answer="RAG combines retrieval and generation for grounded answers.",
        query="What is RAG?",
        context_used=[retrieved],  # Must be RetrievedChunk, not just Chunk
        model="gpt-4o-mini",
        processing_time_ms=100.0,
        context_token_estimate=50,
    )

    eval_output = EvaluatorOutput(
        overall_consistency_score=0.89,
        is_reliable=True,
        claims=[],
        summary="Output is consistent with context",
        processing_time_ms=50.0,
    )

    pipeline.query = MagicMock(
        return_value=PipelineResult(
            query="What is RAG?",
            answer="RAG combines retrieval and generation for grounded answers.",
            is_reliable=True,
            consistency_score=0.89,
            retrieval=[retrieved],
            guardrail=None,
            generation=gen_output,
            evaluation=eval_output,
            evaluation_status=EvaluationStatus.COMPLETED,
            evaluation_error=None,
            evaluation_deferred=False,
            total_time_ms=250.0,
        )
    )

    # Mock components for health check
    pipeline._embeddings = MagicMock()
    pipeline._embeddings.embed = MagicMock(return_value=__import__("numpy").array([[0.1, 0.2]]))

    pipeline._vector_store = MagicMock()
    pipeline._vector_store.size = 1

    pipeline._llm = MagicMock()
    pipeline._llm._client = True
    pipeline._llm._encoder = True

    pipeline.close = MagicMock()

    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Fixture to provide FastAPI test client with mocked pipeline."""
    previous_auth_disabled = os.environ.get("RAG_AUTH_DISABLED")
    os.environ["RAG_AUTH_DISABLED"] = "true"

    def mock_get_pipeline():
        return mock_pipeline

    app.dependency_overrides[get_pipeline] = mock_get_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    if previous_auth_disabled is None:
        os.environ.pop("RAG_AUTH_DISABLED", None)
    else:
        os.environ["RAG_AUTH_DISABLED"] = previous_auth_disabled


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "version" in data

        # Check components are present
        assert "embeddings" in data["components"]
        assert "vector_store" in data["components"]
        assert "llm" in data["components"]

    def test_health_check_components(self, client):
        """Test health check components are reported correctly."""
        response = client.get("/health")
        data = response.json()

        # All components should report a status
        for component, status in data["components"].items():
            assert status in ["healthy", "degraded", "unhealthy"]


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_metadata(self, client):
        """Test root endpoint returns API metadata."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestIngestEndpoint:
    """Tests for POST /ingest endpoint."""

    def test_ingest_success(self, client):
        """Test successful document ingestion."""
        payload = {
            "texts": [
                "RAG combines retrieval and generation for grounded answers.",
                "Vector databases enable semantic search over embeddings.",
            ],
            "source": "test",
            "metadata": {"test": True},
        }

        response = client.post("/ingest", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["chunks_ingested"] > 0
        assert data["source"] == "test"
        assert "message" in data

    def test_ingest_empty_texts_fails(self, client):
        """Test ingest fails with empty texts."""
        payload = {"texts": [], "source": "test"}

        response = client.post("/ingest", json=payload)

        assert response.status_code == 422  # Validation error

    def test_ingest_whitespace_only_fails(self, client):
        """Test ingest fails with whitespace-only texts."""
        payload = {"texts": ["   ", "\n\t"], "source": "test"}

        response = client.post("/ingest", json=payload)

        assert response.status_code == 422  # Validation error

    def test_ingest_missing_texts_fails(self, client):
        """Test ingest fails without texts field."""
        payload = {"source": "test"}

        response = client.post("/ingest", json=payload)

        assert response.status_code == 422

    def test_ingest_too_many_texts_fails(self, client):
        """Test ingest fails with too many texts."""
        payload = {"texts": ["text"] * 1001, "source": "test"}

        response = client.post("/ingest", json=payload)

        assert response.status_code == 422

    def test_ingest_with_metadata(self, client):
        """Test ingest with custom metadata."""
        payload = {
            "texts": ["Test document"],
            "source": "test-source",
            "metadata": {"type": "test", "version": "1.0", "author": "tester"},
        }

        response = client.post("/ingest", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_ingest_rate_limiting(self, client):
        """Test rate limiting on ingest endpoint (10 requests/minute)."""
        payload = {"texts": ["Test document"], "source": "test"}

        # Make 11 requests - the 11th should be rate limited
        limiter._storage.reset()
        status_codes = []
        for i in range(12):
            response = client.post("/ingest", json=payload)
            status_codes.append(response.status_code)

        assert 200 in status_codes
        assert 429 in status_codes
        assert status_codes.count(200) < len(status_codes)


class TestQueryEndpoint:
    """Tests for POST /query endpoint."""

    def test_query_empty_store_handles_gracefully(self, client):
        """Test query on empty vector store returns appropriate response."""
        # Clear the vector store first by creating a fresh pipeline
        payload = {"query": "test question?", "top_k": 5, "sync_evaluation": True}

        response = client.post("/query", json=payload)

        # Should still return 200 with fallback answer
        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert "answer" in data
        assert "is_reliable" in data
        assert "consistency_score" in data
        assert "retrieved_chunks" in data
        assert "processing_time_ms" in data

    def test_query_basic(self, client):
        """Test basic query operation."""
        # First ingest some data
        ingest_payload = {"texts": ["RAG systems combine retrieval and generation."], "source": "test"}
        client.post("/ingest", json=ingest_payload)

        # Now query
        query_payload = {"query": "What is RAG?", "top_k": 5}

        response = client.post("/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "What is RAG?"
        assert "answer" in data
        assert data["is_reliable"] is not None
        assert data["consistency_score"] >= 0
        assert isinstance(data["retrieved_chunks"], list)
        assert data["processing_time_ms"] > 0

    def test_query_with_top_k(self, client):
        """Test query with custom top_k parameter."""
        query_payload = {"query": "test question", "top_k": 3}

        response = client.post("/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()

        # Should have at most top_k chunks
        assert len(data["retrieved_chunks"]) <= 3

    def test_query_with_sync_evaluation(self, client):
        """Test query with synchronous evaluation."""
        query_payload = {"query": "test question", "sync_evaluation": True}

        response = client.post("/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()

        # With sync evaluation, status should be COMPLETED or PENDING
        assert data["evaluation_status"] in ["COMPLETED", "PENDING", "FAILED"]

    def test_query_streaming_response(self, client):
        """Test streaming query response."""
        # First ingest some data
        ingest_payload = {"texts": ["RAG systems work by retrieving and generating."], "source": "test"}
        client.post("/ingest", json=ingest_payload)

        query_payload = {"query": "How do RAG systems work?", "stream": True}

        response = client.post("/query", json=query_payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # Parse streaming response
        events = []
        for line in response.iter_lines():
            if line:
                events.append(json.loads(line))

        assert len(events) > 0

        # Should have at least a result event
        result_events = [e for e in events if e.get("type") == "result"]
        assert len(result_events) > 0

    def test_query_missing_query_fails(self, client):
        """Test query fails without query field."""
        payload = {"top_k": 5}

        response = client.post("/query", json=payload)

        assert response.status_code == 422

    def test_query_empty_query_fails(self, client):
        """Test query fails with empty query string."""
        payload = {"query": "", "top_k": 5}

        response = client.post("/query", json=payload)

        assert response.status_code == 422

    def test_query_very_long_query_fails(self, client):
        """Test query fails with query exceeding max length."""
        payload = {"query": "x" * 2001, "top_k": 5}

        response = client.post("/query", json=payload)

        assert response.status_code == 422

    def test_query_invalid_top_k_fails(self, client):
        """Test query fails with invalid top_k."""
        payload = {"query": "test", "top_k": 0}  # Invalid: must be >= 1

        response = client.post("/query", json=payload)

        assert response.status_code == 422

    def test_query_top_k_too_high_fails(self, client):
        """Test query fails with top_k > 100."""
        payload = {"query": "test", "top_k": 101}  # Invalid: max is 100

        response = client.post("/query", json=payload)

        assert response.status_code == 422

    def test_query_rate_limiting(self, client):
        """Test rate limiting on query endpoint (30 requests/minute)."""
        payload = {"query": "test question", "top_k": 5}

        # Make multiple requests to test rate limiting
        limiter._storage.reset()
        status_codes = []
        for i in range(32):
            response = client.post("/query", json=payload)
            status_codes.append(response.status_code)

        assert 200 in status_codes
        assert 429 in status_codes
        assert status_codes.count(200) < len(status_codes)


class TestResponseModels:
    """Tests for response model validation."""

    def test_ingest_response_structure(self, client):
        """Test IngestResponse model matches response."""
        payload = {"texts": ["test document"], "source": "test"}

        response = client.post("/ingest", json=payload)
        data = response.json()

        # Validate expected fields
        required_fields = {"success", "chunks_ingested", "source", "message"}
        assert required_fields.issubset(data.keys())

    def test_query_response_structure(self, client):
        """Test QueryResponse model matches response."""
        payload = {"query": "test question"}

        response = client.post("/query", json=payload)
        data = response.json()

        # Validate expected fields
        required_fields = {
            "query",
            "answer",
            "is_reliable",
            "consistency_score",
            "retrieved_chunks",
            "processing_time_ms",
            "evaluation_status",
        }
        assert required_fields.issubset(data.keys())

    def test_retrieved_chunk_structure(self, client):
        """Test RetrievedChunkResponse model structure."""
        # Ingest and query to get chunks
        ingest_payload = {"texts": ["Test document for retrieval"], "source": "test"}
        client.post("/ingest", json=ingest_payload)

        query_payload = {"query": "test", "top_k": 5}

        response = client.post("/query", json=query_payload)
        data = response.json()

        if data["retrieved_chunks"]:
            chunk = data["retrieved_chunks"][0]

            # Validate chunk fields
            required_chunk_fields = {"content", "source", "chunk_index", "similarity_score"}
            assert required_chunk_fields.issubset(chunk.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
