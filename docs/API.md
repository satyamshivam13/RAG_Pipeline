# RAG Pipeline FastAPI Documentation

## Quick Start

### 1. Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export RAG_API_TOKEN="your-optional-secret-token"

# Run the API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 2. Docker Compose (Recommended for Production)

```bash
# Copy example environment
cp .env.example .env
# Edit .env with your credentials

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop the service
docker-compose down
```

### 3. Docker Single Container

```bash
# Build image
docker build -t rag-pipeline:latest .

# Run container with environment variables
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-api-key" \
  -v $(pwd)/vector_store_data:/app/vector_store_data \
  --name rag-api \
  rag-pipeline:latest

# View logs
docker logs -f rag-api

# Stop container
docker stop rag-api
```

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Purpose:** Check the health status of the pipeline and its components.

**Response:**
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

**Example:**
```bash
curl -X GET http://localhost:8000/health
```

---

### 2. Ingest Documents

**Endpoint:** `POST /ingest`

**Purpose:** Ingest documents into the RAG pipeline.

**Rate Limit:** 10 requests per minute

**Headers:**
```
Content-Type: application/json
Authorization: Bearer {RAG_API_TOKEN}  # Optional if RAG_API_TOKEN is set
```

**Request Body:**
```json
{
  "texts": [
    "RAG combines retrieval and generation for grounded answers.",
    "Vector databases enable semantic search over embeddings."
  ],
  "source": "documentation",
  "metadata": {
    "type": "technical_docs",
    "version": "1.0"
  }
}
```

**Response:**
```json
{
  "success": true,
  "chunks_ingested": 4,
  "source": "documentation",
  "message": "Successfully ingested 4 chunks from 2 texts"
}
```

**Examples:**

```bash
# Without authentication
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "RAG combines retrieval and generation.",
      "Vector databases support semantic search."
    ],
    "source": "docs",
    "metadata": {"doc_type": "intro"}
  }'

# With bearer token authentication
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "texts": ["Your document text here"],
    "source": "api"
  }'
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/ingest"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-token"  # Optional
}
payload = {
    "texts": [
        "RAG combines retrieval and generation for grounded answers.",
        "Vector databases enable semantic search over embeddings."
    ],
    "source": "documentation",
    "metadata": {"type": "technical_docs"}
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

### 3. Query the Pipeline

**Endpoint:** `POST /query`

**Purpose:** Query the RAG pipeline and get a grounded answer.

**Rate Limit:** 30 requests per minute

**Headers:**
```
Content-Type: application/json
Authorization: Bearer {RAG_API_TOKEN}  # Optional if RAG_API_TOKEN is set
```

**Request Body:**
```json
{
  "query": "How does RAG reduce hallucination?",
  "top_k": 10,
  "enable_guardrail": false,
  "sync_evaluation": true,
  "stream": false
}
```

**Response (Non-Streaming):**
```json
{
  "query": "How does RAG reduce hallucination?",
  "answer": "RAG reduces hallucination by grounding generation in retrieved evidence...",
  "is_reliable": true,
  "consistency_score": 0.89,
  "retrieved_chunks": [
    {
      "content": "RAG combines retrieval and generation for grounded answers.",
      "source": "documentation",
      "chunk_index": 0,
      "similarity_score": 0.92,
      "metadata": {"type": "technical_docs"}
    }
  ],
  "processing_time_ms": 523.45,
  "evaluation_status": "COMPLETED"
}
```

**Examples:**

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does RAG reduce hallucination?"
  }'

# Advanced query with options
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token" \
  -d '{
    "query": "What are vector databases?",
    "top_k": 5,
    "enable_guardrail": true,
    "sync_evaluation": true,
    "stream": false
  }'

# Streaming response (newline-delimited JSON)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does RAG work?",
    "stream": true
  }'
```

**Python Example (Non-Streaming):**
```python
import requests

url = "http://localhost:8000/query"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-token"  # Optional
}
payload = {
    "query": "How does RAG reduce hallucination?",
    "top_k": 10,
    "sync_evaluation": True
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Reliability: {result['is_reliable']}")
print(f"Consistency Score: {result['consistency_score']:.2f}")
print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
```

**Python Example (Streaming):**
```python
import requests
import json

url = "http://localhost:8000/query"
headers = {"Content-Type": "application/json"}
payload = {
    "query": "How does RAG work?",
    "stream": True
}

with requests.post(url, json=payload, headers=headers, stream=True) as r:
    for line in r.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "result":
                print(f"Answer: {event['answer']}")
            elif event["type"] == "chunk":
                print(f"Retrieved: {event['content'][:100]}...")
            elif event["type"] == "error":
                print(f"Error: {event['error']}")
```

---

## Environment Variables

### Core Configuration

```env
# OpenAI / LLM Configuration
OPENAI_API_KEY=sk-...your-api-key...
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=                          # Optional: for vLLM / Ollama endpoints

# Embeddings Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu                   # cpu | cuda

# Vector Store Configuration
VECTOR_INDEX_TYPE=flat                 # flat | ivf | hnsw
VECTOR_STORE_PERSIST_DIR=./vector_store_data

# API Configuration
RAG_API_TOKEN=your-secret-token        # Optional: bearer token for auth
RAG_AUTH_DISABLED=false                # Set to true to disable auth
CORS_ORIGINS=http://localhost:3000     # Comma-separated list
TRUSTED_HOSTS=localhost,127.0.0.1      # Comma-separated list

# Telemetry Configuration
TELEMETRY_ENABLED=true
TELEMETRY_EXPORTER=console             # console | otlp
TELEMETRY_OTLP_ENDPOINT=               # e.g., http://localhost:4318
TELEMETRY_SERVICE_NAME=rag-pipeline

# Python Configuration
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

---

## Authentication

### Bearer Token Authentication

If `RAG_API_TOKEN` environment variable is set, all `/ingest` and `/query` endpoints require a bearer token:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question"}'
```

To disable authentication, set `RAG_AUTH_DISABLED=true`.

---

## Rate Limiting

Rate limits are applied per client IP:

| Endpoint | Rate Limit | Window |
|----------|-----------|--------|
| `/ingest` | 10 requests | 1 minute |
| `/query` | 30 requests | 1 minute |

When rate limited, you'll receive a 429 (Too Many Requests) response:
```json
{
  "error": "RateLimitExceeded",
  "detail": "Rate limit exceeded: 10 per 1 minute",
  "status_code": 429
}
```

---

## Swagger / OpenAPI Documentation

Once the API is running, access interactive documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`

---

## Error Handling

All errors are returned with a structured response:

```json
{
  "error": "QueryFailed",
  "detail": "Query failed: vector store is empty",
  "status_code": 400,
  "timestamp": 1715710800.123,
  "request_id": "abc-123-def"
}
```

Common status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid payload) |
| 401 | Unauthorized (invalid/missing token) |
| 429 | Rate Limit Exceeded |
| 503 | Service Unavailable (pipeline not initialized) |

---

## Performance Tuning

### Docker Compose

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "4"        # CPU cores
      memory: 8G       # RAM
    reservations:
      cpus: "2"
      memory: 4G
```

### Uvicorn Workers

Increase workers in `Dockerfile` or Docker Compose for higher concurrency:

```bash
# For 4+ core machines (development)
uvicorn api:app --workers 4

# For 8+ core machines with uvloop installed
uvicorn api:app --workers 8 --loop uvloop
```

### Vector Store Index

For large datasets, configure index type in `.env`:

```env
VECTOR_INDEX_TYPE=hnsw     # Best for retrieval speed
# or
VECTOR_INDEX_TYPE=ivf      # Better memory efficiency, faster for large datasets
```

---

## Deployment Examples

### AWS ECS / Fargate

```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URI
docker tag rag-pipeline:latest YOUR_ECR_URI/rag-pipeline:latest
docker push YOUR_ECR_URI/rag-pipeline:latest

# Create ECS task definition with environment variables
# Launch Fargate task with 2 vCPU, 4GB RAM
```

### Heroku / Railway / Render

```bash
# Ensure Dockerfile is in root (done)
# Configure buildpacks for Python
# Set environment variables in platform dashboard
# Deploy via git push
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
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: openai-api-key
        - name: RAG_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: api-token
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

---

## Monitoring & Observability

### Local Observability (Docker Compose with Jaeger)

Uncomment the Jaeger service in `docker-compose.yml` and access the UI at `http://localhost:16686`.

### Structured Logging

Logs are automatically structured with correlation IDs and component names:

```
2024-05-14 12:30:00 - api - INFO - query.start query_length=25 top_k=10 stream=False
2024-05-14 12:30:01 - api - INFO - query.complete answer_length=156 score=0.89 elapsed_ms=523.45
```

---

## Troubleshooting

### Vector Store is Empty

- Run `/ingest` endpoint with sample documents first
- Check that vector_store_data directory has write permissions

### LLM API Errors

- Verify `OPENAI_API_KEY` is set correctly
- Check API quotas and rate limits in OpenAI dashboard
- For local models, ensure `LLM_BASE_URL` points to running server (vLLM, Ollama)

### High Latency

- Increase number of uvicorn workers
- Use HNSW or IVF index for faster retrieval
- Run embedding model on GPU: `EMBEDDING_DEVICE=cuda`

### Out of Memory

- Reduce vector store size or use index quantization
# Reduce `chunking.chunk_size` in `config.py` (see [config.py](config.py#L42)) — lower the `ChunkingConfig.chunk_size` to reduce memory and context usage
- Use IVF index with quantization

---

## API Client Libraries

### Python
Note: The `RAGClient` shown below is a hypothetical wrapper example (not included). You can instead use raw HTTP calls with `requests` as shown in other examples.
```python
from rag_client import RAGClient

client = RAGClient("http://localhost:8000", token="your-token")
client.ingest(["document 1", "document 2"], source="docs")
result = client.query("your question")
print(result.answer)
```

### JavaScript/Node.js
Note: The `RAGClient` shown below is a hypothetical wrapper example (not included). Use raw `fetch`/`axios` calls to the HTTP API in production if you don't have a client library.
```javascript
const client = new RAGClient("http://localhost:8000", "your-token");
await client.ingest(["doc1", "doc2"], "docs");
const result = await client.query("your question");
console.log(result.answer);
```

---

## Support & Contributions

For issues, feature requests, or contributions, please refer to the main repository README and CONTRIBUTING.md.
