# Clean multi-stage Dockerfile
FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /build

# Minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /build/requirements.txt
RUN python -m venv $VIRTUAL_ENV && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /build/requirements.txt

# Copy source (after deps to leverage cache)
COPY . /build/app

# Runtime
FROM python:3.11-slim

LABEL maintainer="RAG Pipeline Team"

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal runtime utilities (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 app && useradd -m -u 1000 -g app app

# Persistent data dir for vectors
RUN mkdir -p /app/vector_store_data && chown -R app:app /app/vector_store_data

# Copy venv and app from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/app /app
# Ensure start.sh is explicitly present (fail fast during build if missing)
COPY --from=builder /build/app/start.sh /app/start.sh
RUN chmod +x /app/start.sh
RUN chown -R app:app /app

USER app

EXPOSE 8000

# Healthcheck using curl which is installed above
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Entrypoint handles startup tasks and launches Gunicorn by default
ENTRYPOINT ["/app/start.sh"]
