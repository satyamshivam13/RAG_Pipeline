# Concerns Map

## High Priority
1. Latency stacking from sequential LLM calls in guardrail + generator + evaluator.
2. Guardrail and evaluator are both request-path blockers.
3. No observable request tracing, making bottleneck/root-cause analysis difficult.

## Retrieval Quality Risks
1. Default embedding model is outdated and low-dimensional.
2. Chunking is mostly fixed-size with weak semantic boundaries.
3. No reranking stage after initial retrieval.

## Reliability Risks
1. Context assembly has no strict token-budget enforcement.
2. FAISS coupling limits backend portability and metadata filtering capabilities.
3. Dimension/index mismatch failure handling is not explicit enough for operators.

## Productization Risks
1. No API service boundary, auth, or rate limiting.
2. No caching layer to reduce repeated query cost.
3. No benchmark-driven evaluation loop in CI.

## Refactor Guidance
- Prioritize stabilization changes that reduce latency and hard failures before broad feature expansion.
- Introduce observability and baseline metrics before aggressive optimization so improvements are measurable.
