# Phase 2: Measure and Observe - Context

**Gathered:** 2026-04-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Make quality and reliability measurable by adding an offline evaluation dataset, RAG metrics, structured logging with correlation IDs, end-to-end tracing spans, and CI quality regression checks.

</domain>

<decisions>
## Implementation Decisions

### Evaluation dataset strategy
- **D-01:** Use a mixed dataset strategy with a hand-labeled core and synthetic expansion.
- **D-02:** Store the evaluation dataset as JSONL in-repo with deterministic fields required for repeatable runs.
- **D-03:** Start with 150 examples for this phase (expand later as coverage grows).

### Metric suite and evaluation command
- **D-04:** Use RAGAS metrics for faithfulness, answer relevancy, context precision, and context recall.
- **D-05:** Provide a single reproducible CLI command for offline evaluation report generation.

### Observability implementation
- **D-06:** Add structured logs with a request correlation ID propagated through query execution.
- **D-07:** Add OpenTelemetry spans for retrieval, generation, and evaluation under a root query span.
- **D-08:** Use console export by default with OTLP-ready configuration via environment variables.

### CI quality gates
- **D-09:** Fail CI when faithfulness is below 0.90.
- **D-10:** Fail CI when answer relevancy is below 0.95.
- **D-11:** Track context precision and context recall in reports now; enforce them as hard gates in a later phase after baseline stabilization.

### the agent's Discretion
- Final JSONL field naming for dataset rows as long as metric inputs are deterministic.
- Trace attribute naming details, while keeping component latency and retrieval metadata visible.
- Report file naming and retention approach for local and CI runs.

</decisions>

<specifics>
## Specific Ideas

- Keep the evaluation run command simple and scriptable for local use and CI.
- Keep observability implementation lightweight for now: low-friction defaults, production-ready wiring paths.
- Preserve current module boundaries; add instrumentation and eval tooling without re-architecting Phase 1 runtime behavior.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project requirements
- `.planning/PROJECT.md` - project value, constraints, and active requirements
- `.planning/REQUIREMENTS.md` - MEAS-01 to MEAS-04 requirement definitions
- `.planning/ROADMAP.md` - Phase 2 goal, deliverables, and exit criteria
- `.planning/STATE.md` - current milestone and progress state

### Architecture and quality guidance
- `RAG_PIPELINE_AUDIT.md` - observability and evaluation gaps to close
- `GOD_TIER_RAG_REBUILD.md` - target direction for evaluation and telemetry
- `.planning/codebase/ARCHITECTURE.md` - current orchestration flow and integration points
- `.planning/codebase/STACK.md` - supported runtime and dependency conventions
- `.planning/codebase/CONVENTIONS.md` - logging and testing patterns
- `.planning/codebase/TESTING.md` - current coverage and benchmark gaps

### Prior phase context
- `.planning/phase-01-stabilize/01-CONTEXT.md` - locked Phase 1 runtime decisions to preserve
- `.planning/phases/01-stabilize/01-01-SUMMARY.md` - deferred evaluator and runtime behavior now in place

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `main.py` already centralizes query orchestration and per-stage timing collection.
- `models.py` already carries runtime metadata fields that can be extended for telemetry metadata.
- `llm_client.py` and stage modules (`retriever.py`, `generator.py`, `evaluator_agent.py`) provide stable insertion points for structured logs and span attributes.

### Established Patterns
- Dataclass config objects in `config.py` are the preferred way to add runtime toggles.
- Pydantic models are the stable boundary contracts between pipeline stages.
- Unit tests in `tests/` are deterministic and mock LLM behavior where practical.

### Integration Points
- Evaluation command should execute pipeline queries through `main.RAGPipeline` to keep metrics aligned with runtime behavior.
- Correlation IDs should be created at query entry in `main.py` and propagated to stage-level logs.
- Tracing spans should wrap retrieval, generation, and evaluation execution paths in existing stage boundaries.

</code_context>

<deferred>
## Deferred Ideas

- Reranking and hybrid retrieval optimization - Phase 3.
- Runtime caching, streaming, and batch interfaces - Phase 3.
- API hardening, auth, rate limiting, and deployment runbooks - Phase 4.
- Hard CI gates for context precision/recall until enough baseline stability data exists.

</deferred>

---

*Phase: 02-measure-observe*
*Context gathered: 2026-04-18*
