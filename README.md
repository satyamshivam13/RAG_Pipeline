<!-- generated-by: gsd-doc-writer -->
# RAG Pipeline

Multi-agent Retrieval Augmented Generation (RAG) pipeline for grounded question answering with retrieval, guardrails, generation, and factual consistency evaluation.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

1. Configure environment variables.

```bash
copy .env.example .env
```

2. Run the interactive demo.

```bash
python demo.py
```

3. Run the test suite.

```bash
pytest tests/ -v
```

## Usage Examples

### Programmatic pipeline usage

```python
from config import PipelineConfig, RuntimeConfig
from main import RAGPipeline

pipeline = RAGPipeline(PipelineConfig(runtime=RuntimeConfig(evaluator_mode="sync")))
pipeline.ingest([
    "Quantum computers use qubits and superposition.",
    "Training large language models requires large compute budgets."
])

result = pipeline.query("How do quantum computers achieve speedups?")
print(result.answer)
print(result.consistency_score)
```

### Ingest text documents from files

```python
from document_loader import DocumentLoader
from config import ChunkingConfig

loader = DocumentLoader(ChunkingConfig())
doc = loader.load_file("notes.txt")
chunks = loader.chunk_documents([doc])
print(len(chunks))
```

## Project Components

- `main.py`: Pipeline orchestration (`RAGPipeline`).
- `retriever.py`: Query embedding plus FAISS search/MMR.
- `guardrail_agent.py`: LLM relevance and safety filtering.
- `generator.py`: Grounded answer generation with context budget.
- `evaluator_agent.py`: Claim-level factual consistency scoring.
- `vector_store.py`: FAISS-backed vector index and persistence.
- `document_loader.py`: Document loading and overlapping chunking.
- `embeddings.py`: Sentence-transformers embedding wrapper.
- `llm_client.py`: OpenAI-compatible client with retries and JSON parsing.

## Running Tests

```bash
pytest tests/ -v
```

## Phase 2 Quality Regression

Run the same flow locally that CI enforces:

```bash
python -m evaluation.run_eval --dataset evaluation/datasets/phase2_eval.jsonl --output evaluation/reports/phase2-latest.json
python -m evaluation.quality_gates --report evaluation/reports/phase2-latest.json
```

Threshold policy:

- faithfulness >= 0.90
- answer_relevancy >= 0.95
- context_precision and context_recall are reported as non-blocking diagnostics in this phase

## License

No LICENSE file is currently present in this repository.
