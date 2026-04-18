# Phase 2: Measure and Observe - Research

**Date:** 2026-04-18
**Phase:** 02-measure-observe

## Summary

Phase 2 should adopt RAGAS for offline quality metrics and OpenTelemetry for trace instrumentation, while keeping deployment friction low through console-default telemetry export and OTLP-ready configuration.

## Findings

### Evaluation stack
- RAGAS supports the required metrics for this phase: faithfulness, answer relevancy, context precision, and context recall.
- A deterministic JSONL dataset plus a single command-line runner is sufficient to satisfy reproducibility goals.
- Recommended initial output format: JSON report containing metric scores, dataset metadata, run timestamp, and gate outcomes.

### Observability stack
- OpenTelemetry Python API/SDK is stable for Python 3.9+ and fits this codebase.
- Manual spans are appropriate for this architecture because pipeline stages are explicit and synchronous/deferred under orchestrator control.
- Correlation IDs should be generated at query entry and attached to logs plus span attributes.

### CI quality gates
- Gate on faithfulness and answer relevancy first to avoid noisy failures while baseline data is still maturing.
- Report context precision and context recall but defer hard-fail enforcement until more historical signal is available.

## Recommended implementation patterns

1. Add an `evaluation/` package:
- `dataset.py` for schema + loading.
- `run_eval.py` for reproducible metric runs.
- `quality_gates.py` for threshold enforcement.

2. Add a `telemetry.py` helper:
- OpenTelemetry tracer provider setup.
- Correlation ID context utilities.
- Structured logging field helpers.

3. Keep defaults safe and local-first:
- Console exporter default.
- OTLP endpoint activation via env/config.

4. Ensure deterministic tests:
- Mock pipeline/LLM calls where possible.
- Validate report schema keys exactly.

## Risks and mitigations

- Risk: CI instability from stochastic LLM outputs.
  - Mitigation: deterministic fixtures/stubs for gate tests; isolate integration runs from strict unit gates.

- Risk: telemetry overhead in query path.
  - Mitigation: minimal span attributes and non-blocking export path.

- Risk: metric drift with small seed dataset.
  - Mitigation: seed with mixed labeled/synthetic coverage and expand in later phases.

## Validation Architecture

### What to validate
- Dataset loader rejects malformed rows and preserves deterministic ordering.
- Eval runner emits required metric keys and report schema every run.
- Correlation IDs propagate through logs and spans.
- Root span includes child spans for retrieval, generation, evaluation.
- Gate command fails on threshold breaches exactly at configured boundaries.

### How to validate
- Unit tests for dataset, telemetry, and gate logic.
- Command-level smoke test for eval run and report generation.
- CI workflow enforcing gate command exit code.

### Pass criteria
- Report includes: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
- CI exits non-zero when faithfulness < 0.90 or answer_relevancy < 0.95.
- Query logs carry correlation_id consistently.
