# RAG Pipeline Rebuild Roadmap

Status: active
Planning date: 2026-04-12
References: RAG_PIPELINE_AUDIT.md, GOD_TIER_RAG_REBUILD.md, .planning/REQUIREMENTS.md

## Planning Goals
- Convert the current educational prototype into a measurable, production-ready RAG system.
- Reduce p95 latency and cost while increasing retrieval and answer quality.
- Keep delivery incremental so each phase ships usable improvements.

## Current Baseline (from audit)
- Overall readiness: 3.4/10
- Latency: ~5s per query (4 sequential LLM calls)
- Evaluation: no ground-truth benchmarking
- Observability: basic logs only
- Retrieval quality risks: outdated embeddings, naive chunking, no reranking

## Target End-State
- Faithfulness >= 0.90
- Answer relevancy >= 0.95
- p95 latency <= 0.8s for warm paths
- Cost per query <= $0.01
- Full tracing + measurable quality gates in CI

## Phase Plan

### Phase 1 - Stabilize Core Pipeline (Week 1-2)
Objective: remove obvious latency/cost bottlenecks and hard failure modes.

Requirements covered:
- PIPE-01, PIPE-02, PIPE-03, PIPE-04, RETR-01

Deliverables:
- Remove LLM guardrail stage from request path; use retrieval score thresholding.
- Decouple evaluator from blocking response path (background evaluation).
- Add context token budget management to prevent overflow.
- Replace fixed character chunking with semantic-aware splitter.
- Upgrade embedding default to a modern model and align dimensions/index.

Exit criteria:
- Median response latency < 2s on demo workload.
- No context-window overflow errors in stress tests.
- All existing tests pass; new tests added for changed behavior.

Plans:
- [ ] 01-01-PLAN.md - Remove blocking guardrail runtime stage and make evaluator deferred by default.
- [ ] 01-02-PLAN.md - Implement semantic chunking and hard context budget truncation behavior.
- [ ] 01-03-PLAN.md - Upgrade embedding default and enforce dimension/index compatibility checks.

### Phase 2 - Measure & Observe (Week 3-4)
Objective: make quality and reliability measurable.

Requirements covered:
- MEAS-01, MEAS-02, MEAS-03, MEAS-04

Deliverables:
- Add offline evaluation dataset (100-300 labeled QA examples).
- Integrate RAGAS metrics (faithfulness, answer relevancy, context precision/recall).
- Add structured logging with request correlation IDs.
- Add tracing spans for retrieval, generation, evaluation.
- Add basic quality regression checks in CI.

Exit criteria:
- Evaluation command produces reproducible metric report.
- Every query trace includes component latency + retrieval metadata.
- CI fails on agreed regression thresholds.

Plans:
- [ ] 02-01-PLAN.md - Build offline dataset and reproducible RAGAS evaluation command/report pipeline.
- [ ] 02-02-PLAN.md - Add structured logging with correlation IDs and OpenTelemetry stage spans.
- [ ] 02-03-PLAN.md - Add CI quality gates for faithfulness and answer relevancy with report artifacts.

### Phase 3 - Optimize Retrieval + Runtime (Week 5-6)
Objective: improve answer quality and runtime efficiency.

Requirements covered:
- RETR-02, RETR-03, PERF-01, PERF-02, PERF-03

Deliverables:
- Add reranking stage (cross-encoder on top candidates).
- Add optional hybrid retrieval (dense + sparse fusion).
- Introduce query and embedding caching.
- Add streaming response mode.
- Add batch query interface for throughput workloads.

Exit criteria:
- p95 latency <= 1s on warm cache path.
- Cost/query reduced by at least 50% from baseline.
- Retrieval quality uplift verified against Phase 2 benchmark.

### Phase 4 - Productionize API Surface (Week 7-8)
Objective: harden for deployment and operations.

Requirements covered:
- PROD-01, PROD-02, PROD-03, PROD-04

Deliverables:
- FastAPI service with async endpoints and streaming endpoint.
- Authentication, rate limiting, and error budget dashboards.
- Containerized deployment artifacts and runbooks.
- SLO monitoring and alert rules.

Exit criteria:
- Production checklist complete.
- Can run load test with stable error rate and SLO adherence.

## Dependency Notes
- Phase 2 depends on stable output format from Phase 1.
- Phase 3 optimization decisions depend on Phase 2 measurements.
- Phase 4 relies on agreed metrics and tracing from Phase 2.

## Immediate Next Action
Execute Phase 2 plans in .planning/phases/02-measure-observe/02-01-PLAN.md, .planning/phases/02-measure-observe/02-02-PLAN.md, and .planning/phases/02-measure-observe/02-03-PLAN.md.

## Completion Definition
Milestone complete when all four phases are implemented and all v1 requirements in `.planning/REQUIREMENTS.md` are marked complete.







