# Conventions Map

## Coding Style
- Module-level docstrings describe intent and flow.
- Class names are explicit by role (Retriever, GuardrailAgent, EvaluatorAgent).
- Configs are frozen dataclasses to avoid runtime mutation.

## Data Contracts
- Pydantic BaseModel used for all inter-stage payloads.
- Similarity scores constrained to 0.0 to 1.0 in public models.

## Error Handling
- LLM requests wrapped with tenacity retries.
- JSON parse failures are logged and raised.
- Limited fallback behavior in some agents (e.g., missing eval entry kept by default in guardrail).

## Logging
- Standard logging module used; mostly info/debug line logs.
- No structured event schema or correlation ids.

## Testing Practices
- Tests use deterministic fixtures where possible.
- Vector similarity tests rely on handcrafted normalized embeddings.
