# Integrations Map

## External Services
- OpenAI-compatible chat completion endpoint via LLM_BASE_URL and OPENAI_API_KEY.

## External Models
- Embeddings default model: all-MiniLM-L6-v2 (sentence-transformers).
- LLM default model from env LLM_MODEL (fallback gpt-4o-mini).

## Storage Integrations
- FAISS local index persisted to filesystem path configured by VectorStoreConfig.persist_dir.
- Sidecar chunk metadata persisted in JSON file.

## Runtime Inputs
- .env values:
  - OPENAI_API_KEY
  - LLM_BASE_URL
  - LLM_MODEL

## Missing Integrations (as gap)
- No tracing backend (LangSmith/OpenTelemetry).
- No external cache (Redis/memcached).
- No API service layer (FastAPI) yet.
