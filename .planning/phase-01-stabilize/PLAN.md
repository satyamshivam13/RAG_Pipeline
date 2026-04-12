# Phase 1 Plan - Stabilize Core Pipeline

Status: ready
Phase window: Week 1-2
References: RAG_PIPELINE_AUDIT.md, GOD_TIER_RAG_REBUILD.md

## Goal
Cut avoidable latency/cost and remove major reliability issues without overhauling the full architecture.

## In Scope
1. Remove blocking guardrail LLM call from runtime path.
2. Make evaluator non-blocking (background or deferred mode).
3. Add context token-budget enforcement.
4. Replace naive fixed chunking with semantic-aware chunking.
5. Upgrade embedding configuration defaults and index compatibility.
6. Add tests for each changed behavior.

## Out of Scope
- Full hybrid retrieval.
- Full observability platform (Phase 2).
- Full API productionization (Phase 4).

## Detailed Work Plan

### Task 1 - Remove Guardrail Runtime Stage
Files likely affected:
- main.py
- guardrail_agent.py (deprecation path)
- config.py
- models.py
- tests/test_guardrail.py (scope reduction or migrate assertions)

Implementation notes:
- Replace guardrail filtering with retrieval-threshold filtering from RetrieverConfig.
- Keep guardrail code temporarily behind a feature flag or mark deprecated to avoid abrupt API breakage.
- Ensure PipelineResult still reports enough metadata for debugging filtered chunks.

Acceptance criteria:
- Query flow executes retrieval -> generation -> evaluator path with no guardrail LLM call.
- End-to-end output remains stable for demo queries.

### Task 2 - Non-Blocking Evaluator
Files likely affected:
- main.py
- evaluator_agent.py
- models.py

Implementation notes:
- Keep query() behavior backward-compatible by returning answer immediately.
- Run evaluator in background thread/task and store result in optional metadata.
- Add a sync fallback mode for tests and deterministic pipelines.

Acceptance criteria:
- Time-to-first-answer is reduced compared with current baseline.
- Evaluator errors never fail the user response path.

### Task 3 - Context Token Budget Management
Files likely affected:
- generator.py
- config.py
- models.py

Implementation notes:
- Add configurable max_context_tokens.
- Estimate tokens with tiktoken when available; fallback heuristic when unavailable.
- Truncate least relevant chunks first.

Acceptance criteria:
- No model context-length exceptions during long-context test.
- Response contains note when truncation happens (for debugging transparency).

### Task 4 - Semantic-Aware Chunking
Files likely affected:
- document_loader.py
- requirements.txt
- config.py
- tests (new + updated)

Implementation notes:
- Introduce recursive semantic splitter with paragraph/sentence separators.
- Keep overlap configurable.
- Preserve existing loader public API.

Acceptance criteria:
- Chunk boundaries align better with sentence/paragraph structure.
- Existing ingestion flows remain compatible.

### Task 5 - Embedding Upgrade and Index Alignment
Files likely affected:
- config.py
- embeddings.py
- vector_store.py
- README.md

Implementation notes:
- Set a modern default embedding model in config.
- Ensure embedding dimension auto-detection or strict validation before index init.
- Add clear error message for dimension mismatch with persisted index.

Acceptance criteria:
- Fresh index build succeeds with new defaults.
- Dimension mismatch errors are explicit and actionable.

### Task 6 - Test & Validation Pass
Files likely affected:
- tests/test_generator.py
- tests/test_vector_store.py
- tests/test_evaluator.py
- tests/test_guardrail.py
- new integration test file

Validation checklist:
- pytest tests -v passes
- demo.py runs without runtime exceptions
- Latency spot-check on 5 sample questions shows meaningful improvement

## Risk Register
1. Backward compatibility risk:
- Mitigation: use config flags and optional fields for migration period.

2. Dependency bloat risk from new chunking/token libs:
- Mitigation: keep libraries minimal and optional with graceful fallback.

3. Async complexity risk in non-async codebase:
- Mitigation: start with safe background execution primitives and explicit error capture.

## Definition of Done
- All in-scope tasks complete.
- Updated docs for changed defaults/behavior.
- Baseline and post-change latency snapshot documented.
- Phase 2 inputs prepared: stable output schema + metric hook points.

## Execution Order
1. Task 1
2. Task 3
3. Task 2
4. Task 4
5. Task 5
6. Task 6

## Runbook Commands
- pytest tests -v
- python demo.py

## Handoff to Phase 2
Produce:
- Latency before/after summary
- List of available runtime metadata fields for metrics collection
- Confirmed hook points for tracing and structured logs
