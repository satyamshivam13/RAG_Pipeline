# Architecture Map

## Current Flow
1. Ingest:
- document_loader chunks text
- embeddings encodes chunks
- vector_store persists vectors + chunk metadata

2. Query:
- retriever retrieves top candidates from vector store
- guardrail_agent performs LLM relevance/safety filtering
- generator composes context prompt and generates answer
- evaluator_agent scores factual consistency
- main returns PipelineResult with per-stage artifacts

## Composition Root
- main.RAGPipeline wires all dependencies in constructor.
- Shared LLMClient is reused by guardrail, generator, evaluator.

## Boundary Models
- models.py defines typed payloads between stages:
  - RetrievedChunk
  - GuardrailOutput
  - GeneratorOutput
  - EvaluatorOutput
  - PipelineResult

## Strengths
- Clear responsibility split per module.
- Strong typed boundaries using pydantic.
- Centralized configuration objects in config.py.

## Architecture Risks
- Sequential 4-stage query path creates high tail latency.
- LLM guardrail and evaluator both add expensive per-query calls.
- No pluggable interfaces for retrieval/index providers.
- No explicit observability layer (tracing/metrics abstraction).
