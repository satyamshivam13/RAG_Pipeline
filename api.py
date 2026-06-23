"""
Production-grade FastAPI service layer for the RAG Pipeline.

Architecture:
  - Async endpoints for high concurrency
  - Pydantic schemas for request/response validation
  - Middleware for auth, rate-limiting, structured logging
  - Health checks for dependencies
  - Streaming support for long-running queries
  - Error handling with structured responses
  - Optional authentication via bearer tokens

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    status,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator, ValidationError
import secrets
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import PipelineConfig
from main import RAGPipeline
from models import RetrievedChunk
from telemetry import (
    add_counter,
    configure_observability,
    get_current_span_context,
    record_histogram,
    reset_correlation_id,
    set_correlation_id,
    set_span_attributes,
    span_context_or_null,
)

# ────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Schemas (Pydantic Models)
# ────────────────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Ingest documents into the RAG pipeline."""

    texts: list[str] = Field(
        ..., min_items=1, max_items=1000, description="List of document texts to ingest"
    )
    source: str = Field(
        default="api", description="Source identifier for these documents"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Optional metadata attached to all documents"
    )

    @validator("texts")
    def texts_not_empty(cls, v):
        if not v or all(not t.strip() for t in v):
            raise ValueError("At least one non-empty text required")
        return v


class IngestResponse(BaseModel):
    """Response for ingest endpoint."""

    success: bool
    chunks_ingested: int
    source: str
    message: str


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk with metadata and similarity score."""

    content: str
    source: str
    chunk_index: int
    similarity_score: float
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    """Query the RAG pipeline."""

    query: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    top_k: Optional[int] = Field(
        default=10, ge=1, le=100, description="Number of chunks to retrieve"
    )
    enable_guardrail: Optional[bool] = Field(
        default=False, description="Enable guardrail checks"
    )
    sync_evaluation: Optional[bool] = Field(
        default=False, description="Wait for evaluation before returning (vs async)"
    )
    stream: Optional[bool] = Field(
        default=False, description="Stream response (newline-delimited JSON)"
    )


class QueryResponse(BaseModel):
    """Response from query endpoint."""

    query: str
    answer: str
    is_reliable: bool
    consistency_score: float
    retrieved_chunks: list[RetrievedChunkResponse]
    processing_time_ms: float
    evaluation_status: str  # "COMPLETED" | "PENDING" | "FAILED"
    warnings: Optional[list[str]] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    timestamp: float
    components: dict[str, str]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    detail: str
    status_code: int
    timestamp: float
    request_id: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# Global State & Initialization
# ────────────────────────────────────────────────────────────────────────────

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)

# Global pipeline instance (shared across requests)
_pipeline: Optional[RAGPipeline] = None

# Configuration
_config: Optional[PipelineConfig] = None

# Authentication token (from env or config)
_auth_token: Optional[str] = os.getenv("RAG_API_TOKEN", None)

logger.info("Auth token configured: %s", "yes" if _auth_token else "no")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.
    Initializes the RAG pipeline on startup and cleans up on shutdown.
    """
    global _pipeline, _config
    logger.info("FastAPI startup: initializing RAG pipeline...")

    try:
        _config = PipelineConfig()
        configure_observability(_config.telemetry)
        _pipeline = RAGPipeline(_config)
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize RAG pipeline: %s", e)
        raise

    yield

    logger.info("FastAPI shutdown: closing RAG pipeline...")
    try:
        if _pipeline:
            _pipeline.close()
        logger.info("RAG pipeline closed cleanly")
    except Exception as e:
        logger.exception("Error during pipeline shutdown: %s", e)


# ────────────────────────────────────────────────────────────────────────────
# Dependency Injection
# ────────────────────────────────────────────────────────────────────────────


async def verify_auth_token(authorization: Optional[str] = Header(None)) -> None:
    """
    Verify bearer token if auth is enabled.
    Can be made optional via environment variable RAG_AUTH_DISABLED=true.
    """
    if os.getenv("RAG_AUTH_DISABLED", "false").lower() == "true":
        return

    if not _auth_token:
        logger.error("Auth token not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    if not authorization:
        logger.warning("Missing authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme; use Bearer",
        )

    # Compare tokens using constant-time comparison to avoid timing attacks
    try:
        credentials_normalized = str(credentials)
        token_normalized = str(_auth_token)
    except Exception:
        credentials_normalized = credentials
        token_normalized = _auth_token

    if not secrets.compare_digest(credentials_normalized, token_normalized):
        logger.warning("Invalid authorization token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


def get_pipeline() -> RAGPipeline:
    """Dependency to get the global pipeline instance."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized",
        )
    return _pipeline


# ────────────────────────────────────────────────────────────────────────────
# Middleware
# ────────────────────────────────────────────────────────────────────────────


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured logging middleware for all requests.
    Logs request method, path, status, and latency.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        set_correlation_id(request_id)
        attributes = {
            "http.request.method": request.method,
            "url.path": request.url.path,
            "http.route": request.url.path,
            "correlation_id": request_id,
        }

        try:
            with span_context_or_null("http.request", attributes, "rag.api") as span:
                response = await call_next(request)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                set_span_attributes(
                    span,
                    {
                        "http.response.status_code": response.status_code,
                        "duration_ms": elapsed_ms,
                    },
                )
                record_histogram(
                    "rag_http_request_latency_ms",
                    elapsed_ms,
                    attributes={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                    },
                )
                add_counter(
                    "rag_http_requests_total",
                    attributes={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                    },
                )
                span_context = get_current_span_context()
                if span_context:
                    response.headers["x-trace-id"] = span_context["trace_id"]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "http.request method=%s path=%s status=%s latency_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
            response.headers["x-request-id"] = request_id
            return response
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_histogram(
                "rag_http_request_latency_ms",
                elapsed_ms,
                attributes={"method": request.method, "path": request.url.path, "status_code": 500},
            )
            add_counter(
                "rag_http_requests_total",
                attributes={"method": request.method, "path": request.url.path, "status_code": 500},
            )
            logger.error(
                "http.error method=%s path=%s latency_ms=%.2f error=%s request_id=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                str(e),
                request_id,
            )
            raise
        finally:
            reset_correlation_id()


# ────────────────────────────────────────────────────────────────────────────
# Health Checks
# ────────────────────────────────────────────────────────────────────────────


async def check_embeddings_health(pipeline: RAGPipeline) -> str:
    """Check if embedding model is loaded and responsive."""
    try:
        # Try to embed a test string
        test_emb = pipeline._embeddings.embed(["health_check"])
        if test_emb.shape[0] == 1 and test_emb.shape[1] > 0:
            return "healthy"
        return "degraded"
    except Exception as e:
        logger.error("Embeddings health check failed: %s", e)
        return "unhealthy"


async def check_vector_store_health(pipeline: RAGPipeline) -> str:
    """Check if vector store is accessible."""
    try:
        size = pipeline._vector_store.size
        logger.info("Vector store size: %s", size)
        return "healthy"  # Even empty store is ok
    except Exception as e:
        logger.error("Vector store health check failed: %s", e)
        return "unhealthy"


async def check_llm_health(pipeline: RAGPipeline) -> str:
    """Check if LLM client can be instantiated (not actually calling LLM to avoid costs)."""
    try:
        if pipeline._llm._client and pipeline._llm._encoder:
            return "healthy"
        return "degraded"
    except Exception as e:
        logger.error("LLM health check failed: %s", e)
        return "unhealthy"


# ────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ────────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware (order matters - add in reverse order of execution)
app.add_middleware(LoggingMiddleware)
_trusted_hosts = [host.strip() for host in os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1").split(",") if host.strip()]
_environment = os.getenv("ENVIRONMENT", "").lower()
if _environment in {"local", "development", "dev", "test", "testing"} and "testserver" not in _trusted_hosts:
    _trusted_hosts.append("testserver")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_trusted_hosts,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return Response(
        json.dumps(
            ErrorResponse(
                error="RateLimitExceeded",
                detail=f"Rate limit exceeded: {exc.detail}",
                status_code=429,
                timestamp=time.time(),
            ).model_dump()
        ),
        status_code=429,
        media_type="application/json",
    )


# ────────────────────────────────────────────────────────────────────────────
# Endpoints: Health
# ────────────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthCheckResponse)
async def health_check(pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    Health check endpoint.
    Returns status of embeddings, vector store, and LLM components.
    """
    components = {
        "embeddings": await check_embeddings_health(pipeline),
        "vector_store": await check_vector_store_health(pipeline),
        "llm": await check_llm_health(pipeline),
    }

    # Determine overall status
    if all(v == "healthy" for v in components.values()):
        status_val = "healthy"
    elif any(v == "unhealthy" for v in components.values()):
        status_val = "unhealthy"
    else:
        status_val = "degraded"

    if status_val != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=HealthCheckResponse(
                status=status_val,
                timestamp=time.time(),
                components=components,
            ).model_dump(),
        )

    return HealthCheckResponse(
        status=status_val,
        timestamp=time.time(),
        components=components,
    )


@app.get("/")
async def root():
    """Root endpoint: API info."""
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ────────────────────────────────────────────────────────────────────────────
# Endpoints: Ingest
# ────────────────────────────────────────────────────────────────────────────


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=200,
    dependencies=[Depends(verify_auth_token)],
)
@limiter.limit("10/minute")
async def ingest(
    request: Request,
    payload: IngestRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Ingest documents into the RAG pipeline.

    Accepts a list of texts, chunks them, embeds them, and stores them in the vector DB.

    Rate limit: 10 requests per minute.
    """
    logger.info(
        "ingest.start num_texts=%s source=%s",
        len(payload.texts),
        payload.source,
    )

    try:
        start_time = time.perf_counter()

        # Ingest documents
        chunks_ingested = await asyncio.to_thread(
            pipeline.ingest,
            payload.texts,
            payload.source,
            payload.metadata,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "ingest.complete chunks=%s source=%s elapsed_ms=%.2f",
            chunks_ingested,
            payload.source,
            elapsed_ms,
        )

        return IngestResponse(
            success=True,
            chunks_ingested=chunks_ingested,
            source=payload.source,
            message=f"Successfully ingested {chunks_ingested} chunks from {len(payload.texts)} texts",
        )

    except (ValueError, ValidationError) as e:
        logger.warning("ingest.client_error source=%s error=%s", payload.source, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ingest request: {str(e)}",
        )
    except Exception as e:
        logger.exception("ingest.error source=%s error=%s", payload.source, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingest failed due to server error",
        )


# ────────────────────────────────────────────────────────────────────────────
# Endpoints: Query
# ────────────────────────────────────────────────────────────────────────────


async def query_stream_generator(
    query: str,
    pipeline: RAGPipeline,
    enable_guardrail: bool,
    sync_evaluation: bool,
    top_k: int,
) -> AsyncGenerator[str, None]:
    """
    Streaming query generator.
    Yields intermediate results as newline-delimited JSON.
    """
    try:
        logger.info("query_stream.start query_length=%s", len(query))

        # Run query in thread to avoid blocking; pass request options through
        result = await asyncio.to_thread(
            pipeline.query,
            query,
            enable_guardrail=enable_guardrail,
            sync_evaluation=sync_evaluation,
            top_k=top_k,
        )

        # Yield initial result
        intermediate = {
            "type": "result",
            "query": result.query,
            "answer": result.answer,
            "is_reliable": result.is_reliable,
            "consistency_score": result.consistency_score,
            "processing_time_ms": result.total_time_ms,
            "evaluation_status": result.evaluation_status.value,
        }
        yield json.dumps(intermediate) + "\n"

        # Yield retrieved chunks
        for chunk in result.retrieval:
            chunk_dict = {
                "type": "chunk",
                "content": chunk.chunk.content,
                "source": chunk.chunk.source,
                "chunk_index": chunk.chunk.chunk_index,
                "similarity_score": chunk.similarity_score,
            }
            yield json.dumps(chunk_dict) + "\n"

        logger.info("query_stream.complete query_length=%s", len(query))

    except Exception as e:
        logger.exception("query_stream.error error=%s", e)
        error_dict = {
            "type": "error",
            "error": str(e),
        }
        yield json.dumps(error_dict) + "\n"


@app.post("/query", dependencies=[Depends(verify_auth_token)])
@limiter.limit("30/minute")
async def query(
    request: Request,
    payload: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Query the RAG pipeline.

    Returns either a structured JSON response or a streaming response (newline-delimited JSON).

    Rate limit: 30 requests per minute.
    """
    logger.info(
        "query.start query_length=%s top_k=%s stream=%s",
        len(payload.query),
        payload.top_k,
        payload.stream,
    )

    try:
        # If streaming, return StreamingResponse
        if payload.stream:
            return StreamingResponse(
                query_stream_generator(
                    payload.query,
                    pipeline,
                    payload.enable_guardrail,
                    payload.sync_evaluation,
                    payload.top_k or 10,
                ),
                media_type="application/x-ndjson",
            )

        # Otherwise, return full response
        start_time = time.perf_counter()

        # Run query in thread pool to avoid blocking; pass request options through
        result = await asyncio.to_thread(
            pipeline.query,
            payload.query,
            enable_guardrail=payload.enable_guardrail,
            sync_evaluation=payload.sync_evaluation,
            top_k=(payload.top_k or 10),
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "query.complete query_length=%s answer_length=%s score=%.2f elapsed_ms=%.2f",
            len(payload.query),
            len(result.answer),
            result.consistency_score,
            elapsed_ms,
        )

        # Convert retrieved chunks to response format
        retrieved_chunks = [
            RetrievedChunkResponse(
                content=chunk.chunk.content,
                source=chunk.chunk.source,
                chunk_index=chunk.chunk.chunk_index,
                similarity_score=chunk.similarity_score,
                metadata=chunk.chunk.metadata,
            )
            for chunk in result.retrieval
        ]

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            is_reliable=result.is_reliable,
            consistency_score=result.consistency_score,
            retrieved_chunks=retrieved_chunks,
            processing_time_ms=result.total_time_ms,
            evaluation_status=result.evaluation_status.value,
            warnings=result.generation.warnings if result.generation else None,
        )

    except HTTPException:
        # Re-raise HTTPExceptions unchanged
        raise
    except (ValueError, ValidationError, TypeError) as e:
        logger.warning("query.client_error query_length=%s error=%s", len(payload.query), e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query request: {str(e)}",
        )
    except Exception as e:
        logger.exception("query.error query_length=%s error=%s", len(payload.query), e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ────────────────────────────────────────────────────────────────────────────
# OpenAPI Schema Customization
# ────────────────────────────────────────────────────────────────────────────


def custom_openapi():
    """Customize OpenAPI schema for better documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="RAG Pipeline API",
        version="1.0.0",
        description="Production-grade Retrieval-Augmented Generation API",
        routes=app.routes,
    )

    # Add security scheme for bearer token
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token for API authentication (set RAG_API_TOKEN env var)",
        }
    }

    # Add examples to paths
    openapi_schema["paths"]["/ingest"]["post"]["examples"] = {
        "application/json": {
            "summary": "Ingest sample documents",
            "value": {
                "texts": [
                    "RAG combines retrieval and generation for grounded answers.",
                    "Vector databases enable semantic search over embeddings.",
                ],
                "source": "documentation",
                "metadata": {"type": "technical_docs"},
            },
        }
    }

    openapi_schema["paths"]["/query"]["post"]["examples"] = {
        "application/json": {
            "summary": "Query sample",
            "value": {
                "query": "How does RAG reduce hallucination?",
                "top_k": 5,
                "enable_guardrail": False,
                "sync_evaluation": True,
                "stream": False,
            },
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
