# Stack Map

## Language and Runtime
- Python 3.10+
- Synchronous execution model (no asyncio in pipeline path)

## Core Libraries
- openai: LLM API client
- sentence-transformers: embedding model wrapper
- faiss-cpu: vector indexing and ANN primitives
- pydantic v2: schema and runtime validation
- tiktoken: token counting utility
- tenacity: retry/backoff logic
- python-dotenv: env loading
- pytest: test runner

## AI Architecture Pattern
- Multi-agent linear pipeline:
  1. Retriever
  2. Guardrail Agent
  3. Generator Agent
  4. Evaluator Agent

## Notable Constraints
- Retrieval and generation are synchronous; latency compounds stage-by-stage.
- FAISS is directly embedded as storage layer (no abstract vector backend interface).
- All components are local classes composed by main orchestrator.
