# RAG Pipeline Rebuild

## What This Is

This project upgrades the current educational RAG implementation into a production-ready, measurable system for an internal engineering team. It preserves the clean modular foundations already in the repo while rebuilding weak points in retrieval quality, latency, evaluation, and operations. The build plan follows the audit findings and the god-tier architecture target in incremental phases.

## Core Value

Deliver trustworthy, fast, and measurable grounded answers from internal knowledge with clear evidence and operational visibility.

## Requirements

### Validated

- ✓ Modular multi-stage RAG baseline exists (retrieval, guardrail, generation, evaluation) — existing
- ✓ FAISS-backed vector retrieval with persistence exists — existing
- ✓ Unit-test baseline for core components exists — existing

### Active

- [ ] Reduce end-to-end latency and remove avoidable blocking stages in query path
- [ ] Introduce benchmarked evaluation with reproducible quality metrics
- [ ] Improve retrieval quality (semantic chunking, upgraded embeddings, reranking/hybrid retrieval)
- [ ] Add observability (structured logs, tracing, quality telemetry)
- [ ] Add runtime optimization (caching, streaming, batching)
- [ ] Expose hardened API surface with authentication/rate limiting/deployment readiness

### Out of Scope

- Multi-tenant enterprise admin portal in v1 — not required for internal delivery
- Full UI product surface in v1 — API and pipeline quality are the priority
- Fine-tuning proprietary LLMs in v1 — optimize retrieval/prompting first

## Context

- Repository is brownfield Python RAG code with clean component boundaries.
- Current architecture performs sequential LLM calls that raise latency and cost.
- Two reference documents drive this rebuild:
  - RAG_PIPELINE_AUDIT.md (gap analysis and remediation priorities)
  - GOD_TIER_RAG_REBUILD.md (target architecture and implementation direction)
- Primary v1 users are internal engineering teams.
- Success expectation selected by user: complete all four roadmap phases.

## Constraints

- **Compatibility**: Preserve existing Python module structure where practical — reduce migration risk
- **Quality Gate**: Every phase must be verifiable with tests/metrics — avoid subjective progress
- **Performance**: Runtime changes must be benchmarked against current baseline — prevent hidden regressions
- **Operational Safety**: No destructive schema/data changes without rollback path — protect ongoing development

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Brownfield-first modernization | Existing architecture is usable foundation; faster than full rewrite | ✓ Good |
| Phase-based delivery (Stabilize -> Measure -> Optimize -> Productionize) | Enables measurable improvements and controlled risk | ✓ Good |
| Internal engineering team as primary v1 user | Matches immediate adoption path and feedback loop | ✓ Good |
| Milestone completion target is full 4-phase roadmap | Explicit user definition of done | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after initialization*
