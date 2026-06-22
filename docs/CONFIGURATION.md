<!-- generated-by: gsd-doc-writer -->
# Configuration

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Conditionally | `None` | API key consumed by `LLMConfig.api_key`. Required for hosted providers such as OpenAI or Groq. |
| `LLM_BASE_URL` | Optional | `None` | Optional OpenAI-compatible base URL (`https://api.openai.com/v1`, Groq endpoint, or compatible proxy). |
| `LLM_MODEL` | Optional | `gpt-4o-mini` | Default model used by guardrail/generator/evaluator and `LLMClient` unless overridden in code. |
| `TELEMETRY_ENABLED` | Optional | `true` | Enables OpenTelemetry setup and structured logging. |
| `TELEMETRY_EXPORTER` | Optional | `console` | Export mode: `console`, `otlp`, or `none`. |
| `TELEMETRY_OTLP_ENDPOINT` | Optional | `None` | OTLP HTTP endpoint, usually `http://otel-collector:4318`. |
| `TELEMETRY_SERVICE_NAME` | Optional | `rag-pipeline` | Service name attached to traces and metrics. |
| `METRICS_ENABLED` | Optional | `true` | Enables OpenTelemetry metric instruments. |
| `STRUCTURED_LOGS_ENABLED` | Optional | `true` | Emits JSON logs with correlation, trace, and span IDs. |
| `OTEL_METRIC_EXPORT_INTERVAL_MS` | Optional | `60000` | Metric export interval in milliseconds. |
| `ENVIRONMENT` | Optional | `local` | Deployment environment resource attribute. |
| `SERVICE_VERSION` | Optional | `1.0.0` | Service version resource attribute. |
| `LOG_LEVEL` | Optional | `INFO` | Root structured log level. |

## Config File Format

The project uses Python dataclasses in `config.py` as the primary configuration format.

- `EmbeddingConfig`: model choice, dimension, device, batching.
- `VectorStoreConfig`: FAISS index type and persistence settings.
- `RetrieverConfig`: top-k, threshold, MMR behavior.
- `ChunkingConfig`: token-aware chunk size, overlap, strategy, tokenizer, and minimum tail length.
- `LLMConfig`: provider endpoint and retry/timeouts.
- `GuardrailConfig`, `GeneratorConfig`, `EvaluatorConfig`: stage-level model settings.
- `RuntimeConfig`: stage toggles (`use_guardrail`) and evaluator execution mode.
- `TelemetryConfig`: OpenTelemetry exporter, metrics, structured logs, service version, and environment.

## Required vs Optional Settings

- Required for hosted inference: `OPENAI_API_KEY` must be provided for OpenAI/Groq style endpoints.
- Optional with defaults: all other values in `config.py` use dataclass defaults when unset.
- Runtime mode controls:
  - `RuntimeConfig.use_guardrail` defaults to `False`.
  - `RuntimeConfig.evaluator_mode` defaults to `deferred`.

## Defaults

| Setting | Default |
|---|---|
| `EmbeddingConfig.model_name` | `BAAI/bge-large-en-v1.5` |
| `EmbeddingConfig.dimension` | `1024` |
| `EmbeddingConfig.device` | `cpu` |
| `VectorStoreConfig.index_type` | `flat` |
| `VectorStoreConfig.persist_dir` | `./vector_store_data` |
| `RetrieverConfig.top_k` | `10` |
| `RetrieverConfig.similarity_threshold` | `0.3` |
| `RetrieverConfig.use_mmr` | `True` |
| `RetrieverConfig.mmr_top_k` | `5` |
| `ChunkingConfig.chunk_size` | `512` tokens |
| `ChunkingConfig.chunk_overlap` | `64` tokens |
| `ChunkingConfig.min_chunk_size` | `50` tokens |
| `ChunkingConfig.strategy` | `semantic` |
| `ChunkingConfig.tokenizer_model` | `gpt-4o-mini` |
| `ChunkingConfig.tokenizer_encoding` | `cl100k_base` |
| `LLMConfig.default_model` | `gpt-4o-mini` |
| `GuardrailConfig.relevance_threshold` | `0.6` |
| `GeneratorConfig.max_context_tokens` | `3000` |
| `EvaluatorConfig.consistency_threshold` | `0.7` |

## Per-Environment Overrides

- Create a local `.env` file from `.env.example` for developer machine values.
- Override defaults via environment variables (for example, `LLM_MODEL`) per shell/session.
- For test-only behavior, prefer `PipelineConfig` overrides in tests (see `tests/test_pipeline_phase1_runtime.py`) rather than mutating global config files.

## Chunking Strategies

- `semantic`: default RAG strategy. Packs paragraphs and sentences into token-budgeted chunks with bounded token overlap and per-chunk quality metrics.
- `token`: fast sliding token window when strict throughput matters more than semantic boundaries.
- `legacy_char`: backward-compatible character splitter for rollback and before/after comparison.

Chunk quality metrics are stored on each chunk under `metadata["chunking"]["metrics"]`, including token count, sentence count, paragraph count, overlap tokens, budget utilization, semantic-boundary flags, and oversize status.

<!-- VERIFY: Production deployment secret-management location is not discoverable from repository files. -->
