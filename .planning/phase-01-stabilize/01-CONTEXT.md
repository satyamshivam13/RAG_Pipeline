# Phase 1: Stabilize Core Pipeline - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove obvious latency and cost bottlenecks in the current RAG query path without changing the overall product direction. This phase stabilizes the existing pipeline by removing the blocking guardrail from the runtime path, deferring evaluation, enforcing context budgets, improving chunking, and upgrading the embedding default.

</domain>

<decisions>
## Implementation Decisions

### Guardrail removal strategy
- **D-01:** Keep the guardrail module for compatibility/deprecation, but remove it from the default runtime query path.
- **D-02:** Use retrieval-score filtering as the primary runtime gate instead of an LLM guardrail call.

### Evaluator execution mode
- **D-03:** Run evaluation in the background by default so answers return immediately.
- **D-04:** Preserve a synchronous fallback mode for tests and debugging.

### Embedding upgrade choice
- **D-05:** Set the default embedding model to `BAAI/bge-large-en-v1.5`.
- **D-06:** Enforce explicit dimension/index compatibility so mismatches fail clearly instead of degrading silently.

### Chunking and token budget policy
- **D-07:** Replace naive fixed-size chunking with semantic chunking that respects sentence and paragraph boundaries.
- **D-08:** Apply a hard context token budget in generation and truncate the least relevant chunks when the budget is exceeded.
- **D-09:** Emit a clear warning when context is truncated so the behavior is visible during debugging.

</decisions>

<specifics>
## Specific Ideas

- Preserve the existing `guardrail_agent.py` module as a compatibility/deprecation path rather than deleting it immediately.
- Use the current `RetrieverConfig.similarity_threshold` path as the runtime replacement for the guardrail gate.
- The embedding upgrade should align with the audit recommendation to move away from `all-MiniLM-L6-v2` and its 384-dimension constraint.
- Context truncation should be transparent rather than silent so long-document failures are easier to diagnose.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and roadmap
- `.planning/PROJECT.md` — project vision, core value, constraints, and active requirements
- `.planning/REQUIREMENTS.md` — v1 requirements and phase traceability
- `.planning/ROADMAP.md` — phase sequencing and completion definition
- `.planning/phase-01-stabilize/PLAN.md` — detailed Phase 1 task scope and order

### Audit and target architecture
- `RAG_PIPELINE_AUDIT.md` — gap analysis, Phase 1 stabilization rationale, and critical blockers
- `GOD_TIER_RAG_REBUILD.md` — target architecture, retrieval/generation optimization direction, and later-phase goals

### Existing codebase guidance
- `.planning/codebase/STACK.md` — runtime, dependencies, and architecture constraints
- `.planning/codebase/STRUCTURE.md` — module layout and integration points
- `.planning/codebase/CONVENTIONS.md` — data contracts, error handling, and testing patterns

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `main.RAGPipeline`: existing composition root for pipeline changes.
- `retriever.Retriever`: already uses `RetrieverConfig.similarity_threshold`, which can replace guardrail behavior.
- `llm_client.LLMClient`: reusable chat/JSON client with retries and token counting.
- `models.PipelineResult` and related Pydantic models: stable stage boundary contracts for phase changes.
- `vector_store.VectorStore`: current retrieval storage and search implementation.

### Established Patterns
- Frozen dataclass configs in `config.py` are the established way to expose tunables.
- Pydantic models define boundary objects between stages.
- The pipeline is synchronous end-to-end today, so Phase 1 should minimize disruption while introducing deferred work carefully.
- Logging is already present via the standard library; Phase 1 should preserve that shape even if later phases improve observability.

### Integration Points
- `main.py` orchestrates retrieval, guardrail, generation, and evaluation, so the query path changes connect there first.
- `document_loader.py` controls chunk creation and is the insertion point for semantic splitting.
- `generator.py` is where token budget enforcement and truncation behavior belong.
- `evaluator_agent.py` is where the background/sync split is introduced.
- `config.py` and `vector_store.py` must stay aligned with the new embedding dimension.
- `tests/` provide the current validation baseline and will absorb the Phase 1 behavior changes.

</code_context>

<deferred>
## Deferred Ideas

- Full observability platform with structured logging and tracing — Phase 2.
- Offline benchmark dataset and quality metrics loop — Phase 2.
- Reranking and hybrid retrieval — Phase 3.
- Caching, streaming responses, and batch query APIs — Phase 3.
- FastAPI wrapper, authentication, rate limiting, and deployment hardening — Phase 4.
- Multi-hop/graph retrieval and active learning loops — later phases.

</deferred>

---

*Phase: 01-stabilize*
*Context gathered: 2026-04-12*
