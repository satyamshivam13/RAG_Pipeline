# FastAPI Service Implementation Guide

## Overview

This document provides detailed deployment and usage instructions for the RAG Pipeline FastAPI service.

## ✅ What Was Implemented

### 1. **Production-Grade FastAPI Service** (`api.py`)
- **Async endpoints** for high concurrency
- **Health check endpoints** with component diagnostics
- **Request/response validation** using Pydantic models
- **Authentication** with optional bearer token support
- **Rate limiting** (10 req/min for `/ingest`, 30 req/min for `/query`)
- **Streaming support** for long-running queries
- **Middleware stack** with logging, CORS, and security
- **Error handling** with structured responses
- **OpenAPI/Swagger** documentation auto-generation

### 2. **Containerization** 
- **Multi-stage Dockerfile** (optimized size and security)
- **Docker Compose** for local development and testing
- **Non-root user** for security
- **Health checks** for orchestration platforms
- **Volume mounts** for persistent vector store data

### 3. **Comprehensive Documentation**
- **API.md** - Complete endpoint reference with examples
- **.env.example** - All configuration options
- **API tests** - Unit tests for all endpoints

### 4. **Configuration**
- **Environment variables** for all settings
- **Optional authentication** (bearer token)
- **CORS support** for web clients
- **Configurable resource limits**

---

## 🚀 Quick Start

### Option 1: Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export RAG_API_TOKEN="optional-secret-token"

# Run the API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Access Swagger UI: http://localhost:8000/docs

### Option 2: Docker Compose (Recommended)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your credentials

# Start service
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop service
docker-compose down
```

### Option 3: Docker Single Container

```bash
# Build image
docker build -t rag-pipeline:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -v $(pwd)/vector_store_data:/app/vector_store_data \
  --name rag-api \
  rag-pipeline:latest

# Test health
curl http://localhost:8000/health
```

---

## 📡 API Endpoints

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1715710800.123,
  "components": {
    "embeddings": "healthy",
    "vector_store": "healthy",
    "llm": "healthy"
  },
  "version": "1.0.0"
}
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "texts": [
      "RAG combines retrieval and generation.",
      "Vector databases enable semantic search."
    ],
    "source": "documentation",
    "metadata": {"type": "docs"}
  }'
```

### Query Pipeline

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "query": "How does RAG reduce hallucination?",
    "top_k": 10,
    "sync_evaluation": true,
    "stream": false
  }'
```

### Streaming Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "query": "Your question?",
    "stream": true
  }' | jq -R 'fromjson?'
```

---

## 🔐 Authentication

### Enable Bearer Token Auth

```bash
# Set the token environment variable
export RAG_API_TOKEN="your-secret-token"

# All requests must include Authorization header
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Disable Auth

```bash
export RAG_AUTH_DISABLED=true
# No Authorization header required
```

---

## ⚙️ Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...

# API Configuration
RAG_API_TOKEN=optional-bearer-token        # Leave empty to disable
RAG_AUTH_DISABLED=false                    # Set to true to disable auth
CORS_ORIGINS=http://localhost:3000         # Comma-separated
TRUSTED_HOSTS=localhost,127.0.0.1          # Comma-separated

# LLM Configuration
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=                              # Optional: for local models

# Embeddings
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu                       # cpu | cuda

# Vector Store
VECTOR_INDEX_TYPE=flat                     # flat | ivf | hnsw
VECTOR_STORE_PERSIST_DIR=./vector_store_data

# Telemetry
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=console                 # console | otlp
TELEMETRY_OTLP_ENDPOINT=                   # e.g., http://localhost:4318
```

---

## 📊 Rate Limiting

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/ingest` | 10 requests | 1 minute |
| `/query` | 30 requests | 1 minute |

Rate limit is per-client IP. When exceeded, you'll receive a 429 response.

---

## 🧪 Testing

### Run Core Tests

```bash
pytest tests/ --ignore=tests/test_api.py -v
```

### Run API Tests (requires OPENAI_API_KEY)

```bash
export OPENAI_API_KEY="your-key"
pytest tests/test_api.py -v
```

### Test Health Endpoint

```bash
curl -i http://localhost:8000/health
```

### Test Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Test document"],
    "source": "test"
  }'
```

### Test Query on Empty Store

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

---

## 📈 Production Deployment

### AWS ECS/Fargate

```bash
# Push to ECR
aws ecr get-login-password --region YOUR_AWS_REGION | docker login --username AWS --password-stdin YOUR_ECR_URI
docker tag rag-pipeline:latest YOUR_ECR_URI/rag-pipeline:latest
docker push YOUR_ECR_URI/rag-pipeline:latest

# Create ECS task definition with:
# - 2 vCPU, 4GB RAM minimum
# - Environment variables for OPENAI_API_KEY, RAG_API_TOKEN
# - CloudWatch logging
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-pipeline-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-pipeline:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: vector-store
          mountPath: /app/vector_store_data
          readOnly: false
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: vector-store
        persistentVolumeClaim:
          claimName: rag-vector-store-pvc
```

### Heroku/Railway/Render

1. Push repository to GitHub
2. Connect platform to repo
3. Set environment variables in platform dashboard:
   - `OPENAI_API_KEY`
   - `RAG_API_TOKEN` (optional)
4. Platform auto-detects `Dockerfile` and deploys

---

## 🔧 Performance Tuning

### Increase Workers (Docker)

Update `Dockerfile` CMD:
```dockerfile
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "8",      # Increase for more CPU cores
     "--loop", "uvloop"]
```

### Use GPU for Embeddings

```bash
export EMBEDDING_DEVICE=cuda
# Build Docker image with CUDA base

docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04 -t rag-pipeline-gpu .
```

### Optimize Vector Index

```bash
# For large datasets, use HNSW (faster retrieval)
export VECTOR_INDEX_TYPE=hnsw

# For memory-constrained environments, use IVF
export VECTOR_INDEX_TYPE=ivf
```

### Scale Horizontally

```bash
# To scale locally with Docker Compose v1/legacy standalone mode:
docker-compose up -d --scale rag-api=3

# Note: Expose a single port per host. For multiple replicas behind a single host port,
# use a reverse proxy/load balancer (e.g., nginx, Traefik) to distribute traffic to the
# running containers or deploy to a platform that supports service discovery/load balancing.
```

---

## 📋 Monitoring & Logging

### View Logs

```bash
# Docker Compose
docker-compose logs -f rag-api

# Docker
docker logs -f rag-api

# Kubernetes
kubectl logs -f deployment/rag-pipeline-api
```

### Structured Logging

Logs follow this format:
```
2024-05-14 12:30:00 - api - INFO - query.start query_length=25 top_k=10
2024-05-14 12:30:01 - api - INFO - query.complete answer_length=156 score=0.89 elapsed_ms=523.45
```

### OpenTelemetry Export

Enable OTLP export for Jaeger/Datadog:

```bash
export TELEMETRY_EXPORTER=otlp
export TELEMETRY_OTLP_ENDPOINT=http://your-otel-collector:4318

# Or with Jaeger locally
docker run -d \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

---

## 🐛 Troubleshooting

### Vector Store is Empty

```bash
# Ingest sample documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Sample document text"],
    "source": "test"
  }'
```

### LLM API Errors

```bash
# Check OpenAI credentials
export OPENAI_API_KEY="your-valid-key"

# For local models, ensure LLM_BASE_URL is set
export LLM_BASE_URL=http://localhost:8001/v1
```

### High Latency

```bash
# Increase workers
uvicorn api:app --workers 8 --loop-impl uvloop

# Use faster index
export VECTOR_INDEX_TYPE=hnsw

# Run embeddings on GPU
export EMBEDDING_DEVICE=cuda
```

### Rate Limited

- Wait 1 minute or use a different client IP
- Or increase rate limits in `api.py`:
  ```python
  @limiter.limit("100/minute")  # Increase from 30/minute
  ```

---

## 📚 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🧬 Python Client Example

```python
import requests
import json
from typing import List

class RAGClient:
    def __init__(self, base_url: str, token: str = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
    
    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def health(self):
        """Check API health."""
        r = requests.get(f"{self.base_url}/health")
        return r.json()
    
    def ingest(self, texts: List[str], source: str = "api"):
        """Ingest documents."""
        payload = {"texts": texts, "source": source}
        r = requests.post(
            f"{self.base_url}/ingest",
            json=payload,
            headers=self._headers()
        )
        return r.json()
    
    def query(self, query: str, top_k: int = 10, stream: bool = False):
        """Query the pipeline."""
        payload = {"query": query, "top_k": top_k, "stream": stream}
        r = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers=self._headers()
        )
        
        if stream:
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            return r.json()


# Usage
client = RAGClient("http://localhost:8000", token="your-token")

# Health check
print(client.health())

# Ingest
result = client.ingest(["Document text"], source="docs")
print(f"Ingested {result['chunks_ingested']} chunks")

# Query
answer = client.query("Your question?")
print(answer["answer"])
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: API Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ --ignore=tests/test_api.py
      - run: docker build -t rag-pipeline:latest .
```

---

## ✅ Checklist for Production

- [ ] Set `OPENAI_API_KEY` securely (use secrets manager)
- [ ] Set `RAG_API_TOKEN` for API auth
- [ ] Configure `CORS_ORIGINS` for your frontend domain
- [ ] Enable `TELEMETRY_EXPORTER=otlp` for observability
- [ ] Set resource limits in Docker/K8s
- [ ] Configure persistent volumes for `vector_store_data`
- [ ] Set up health checks for load balancers
- [ ] Enable HTTPS/TLS in production
- [ ] Configure autoscaling if needed
- [ ] Set up monitoring/alerting for errors and latency
- [ ] Test rate limiting behavior
- [ ] Test failover and recovery

---

## 📞 Support

For issues or feature requests, refer to the main repository and contribution guidelines.
