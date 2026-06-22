# Observability

The RAG pipeline emits OpenTelemetry traces, metrics, and structured JSON logs. The goal is to debug production issues quickly: slow retrieval, embedding stalls, LLM latency spikes, empty context, vector index behavior, and failed requests.

## Signals

### Traces

- `http.request`: inbound FastAPI request span with method, path, status, latency, request ID, and trace ID response header.
- `rag.ingest`: document ingestion span with document and chunk counts.
- `rag.query`: end-to-end query span with retrieved and filtered counts.
- `rag.retrieve`: retrieval orchestration span.
- `rag.embedding.embed`: embedding batch span with model, text count, batch size, and dimension.
- `rag.vector_store.search` / `rag.vector_store.mmr_search`: FAISS search spans with index type, mode, candidates, and result count.
- `rag.llm.chat`: LLM call span with provider, model, message count, max tokens, latency, and response size.
- `rag.generate` / `rag.evaluate`: generation and evaluation stage spans.

### Metrics

Metrics are exported through OTLP and converted to Prometheus by the example collector.

| Metric | Type | Purpose |
|---|---|---|
| `rag_http_request_latency_ms` | histogram | API latency by method/path/status |
| `rag_http_requests_total` | counter | API traffic and error rate |
| `rag_ingest_latency_ms` | histogram | Ingestion throughput and chunking/embedding/store latency |
| `rag_query_latency_ms` | histogram | End-to-end query latency |
| `rag_retrieval_latency_ms` | histogram | Retrieval orchestration latency |
| `rag_embedding_latency_ms` | histogram | Embedding batch latency by model and text count |
| `rag_llm_latency_ms` | histogram | LLM latency by provider/model |
| `rag_vector_search_latency_ms` | histogram | FAISS search latency by index type and search mode |
| `rag_generation_latency_ms` | histogram | Answer generation latency and truncation behavior |

### Logs

Structured JSON logs include:

- `correlation_id`
- `trace_id`
- `span_id`
- logger, level, timestamp, message
- exception details when present

The API returns `x-request-id` and `x-trace-id` headers so user reports can be joined to traces and logs.

## Configuration

```bash
export TELEMETRY_ENABLED=true
export TELEMETRY_EXPORTER=otlp
export TELEMETRY_OTLP_ENDPOINT=http://localhost:4318
export TELEMETRY_SERVICE_NAME=rag-pipeline
export SERVICE_VERSION=1.0.0
export ENVIRONMENT=local
export METRICS_ENABLED=true
export STRUCTURED_LOGS_ENABLED=true
export OTEL_METRIC_EXPORT_INTERVAL_MS=60000
export LOG_LEVEL=INFO
```

Exporter modes:

- `console`: local development, prints spans and metrics to stdout.
- `otlp`: production mode, exports traces and metrics to an OpenTelemetry Collector.
- `none`: disables exporters while preserving no-op helper behavior.

## Local Stack

Start the collector, Prometheus, Grafana, and Jaeger:

```bash
cd observability
docker compose -f docker-compose.observability.yml up -d
```

Run the API with OTLP export:

```bash
export TELEMETRY_EXPORTER=otlp
export TELEMETRY_OTLP_ENDPOINT=http://localhost:4318
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open:

- Grafana: http://localhost:3000 (`admin` / `admin`)
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686

The Grafana dashboard is provisioned from `observability/grafana/dashboards/rag-pipeline-overview.json`.

## Production Notes

- Keep `BatchSpanProcessor` and periodic metric export enabled; they reduce per-request overhead.
- Prefer low-cardinality metric attributes. This implementation avoids raw query text and document content in telemetry.
- Use logs for incident details and traces for request flow; do not put secrets, prompts, or full retrieved context into span attributes.
- Route OTLP from the app to a local or sidecar collector, then fan out to your vendor backend.
- Alert first on p95 query latency, LLM latency, vector search latency, HTTP 5xx rate, and retrieval result count drops.
