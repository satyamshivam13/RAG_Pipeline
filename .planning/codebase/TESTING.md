# Testing Map

## Existing Coverage Areas
- Vector store search, thresholding, persistence, MMR behavior.
- Guardrail evaluation behavior (unit level).
- Generator behavior (unit level).
- Evaluator behavior (unit level).

## Observed Test Profile
- Primarily unit tests with mocked or controlled components.
- No end-to-end benchmark or regression suite with fixed QA ground truth.
- No latency/cost regression tests.

## Gaps vs Production Goals
- Missing retrieval quality metrics tests (precision@k, recall@k).
- Missing answer quality benchmark dataset and reproducible score pipeline.
- Missing integration tests for full pipeline query path under realistic documents.

## Suggested Near-Term Additions
1. Integration test for main.RAGPipeline query flow with stub LLM responses.
2. Token-budget truncation tests once context budgeting is introduced.
3. Backward-compatibility tests for PipelineResult schema during refactor.
