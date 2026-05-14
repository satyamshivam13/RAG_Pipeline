# Single clean multi-stage Dockerfile for RAG Pipeline
# Builder stage: create a virtualenv and install Python dependencies
FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /build

# Install minimal build dependencies required to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install project dependencies
COPY requirements.txt /build/requirements.txt
RUN python -m venv $VIRTUAL_ENV && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /build/requirements.txt

# Copy application code to leverage cache sensibly (code copy after deps)
COPY . /build/app

# Final runtime image: keep it minimal and secure
FROM python:3.11-slim

LABEL maintainer="RAG Pipeline Team"

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime utilities required for healthcheck and TLS (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and a persistent data directory for vectors
RUN groupadd -r app && useradd --no-log-init -r -g app -m -d /home/app app
RUN mkdir -p /vector_store_data && chown -R app:app /vector_store_data

# Copy virtualenv from builder for reproducible runtime environment
COPY --from=builder /opt/venv /opt/venv

# Copy application files and ensure app user owns them
COPY --chown=app:app . /app

RUN chmod +x /app/start.sh || true

USER app

# Expose FastAPI port
EXPOSE 8000

# Healthcheck used by orchestrators to verify readiness
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Use a small entrypoint that execs the CMD (allows overriding CMD at runtime)
ENTRYPOINT ["/app/start.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "${UVICORN_WORKERS:-1}", "--loop", "uvloop"]

# Notes:
# - Multi-stage build keeps the runtime image small by copying only the virtualenv and app code.
# - The runtime installs only `curl` and `ca-certificates` to support healthchecks and TLS.
# - The container runs as a non-root user `app` for security.
# - The `start.sh` script acts as an entrypoint to allow pre-start tasks and to `exec` the final process.# Multi-stage Dockerfile for RAG_Pipeline
# - Builder stage: installs dependencies into an isolated venv to avoid polluting system Python
# - Runtime stage: copies only the venv and app code to keep image small
# - Uses Python slim base for minimal size and compatibility

FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

# Install system build deps required for some Python packages (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install wheel first to speed up installs
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker cache for dependency installs
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application source
WORKDIR /app
COPY . /app

# ----------------------
# Final runtime image
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

# Create non-root user for security
RUN groupadd -r app && useradd --no-log-init -r -g app -m -d /home/app app

# Create app directory and set ownership
RUN mkdir -p /app /vector_store_data
RUN chown -R app:app /app /vector_store_data

USER app
WORKDIR /app

# Copy venv from builder (only the venv site-packages and bin are needed)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (owned by non-root user)
COPY --chown=app:app . /app

# Expose port for the FastAPI service
EXPOSE 8000

# Healthcheck used by orchestrators to verify the container is ready
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint: use a lightweight start script to allow passing args via docker run/CMD
ENTRYPOINT ["/app/start.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]

# Notes:
# - Using a venv copied from the builder keeps runtime image small and reproducible.
# - We run as a non-root user `app` for security.
# - `uvicorn --workers` is used for moderate concurrency; for heavy workloads consider
#   using Gunicorn with Uvicorn workers and a process manager outside the container.# Multi-stage Dockerfile for RAG Pipeline API
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
