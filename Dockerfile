# Multi-stage Dockerfile for RAG Pipeline API
# Optimized for production with minimal image size and security hardening

# ────────────────────────────────────────────────────────────────────────────
# Builder stage: compile dependencies
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies to a venv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -r requirements.txt

# ────────────────────────────────────────────────────────────────────────────
# Runtime stage: final image
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="RAG Pipeline Team"
LABEL description="Production RAG Pipeline FastAPI Service"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create vector store data directory
RUN mkdir -p /app/vector_store_data && chown -R appuser:appuser /app/vector_store_data

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI application with gunicorn + uvicorn workers for production
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info"]
