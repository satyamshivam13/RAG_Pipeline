# Structure Map

## Root Modules
- main.py: orchestrator and public entrypoint
- config.py: dataclass-based immutable config model
- models.py: pydantic schemas used across stages
- llm_client.py: OpenAI client wrapper with retry + JSON parsing
- embeddings.py: sentence-transformers adapter
- vector_store.py: FAISS index management/search/MMR/persistence
- document_loader.py: text/file loading and chunking
- retriever.py: query embedding + vector search orchestration
- guardrail_agent.py: LLM chunk relevance and safety screening
- generator.py: grounded answer generation from context chunks
- evaluator_agent.py: LLM-based claim consistency scoring
- demo.py: CLI demonstration path

## Tests
- tests/test_vector_store.py
- tests/test_guardrail.py
- tests/test_generator.py
- tests/test_evaluator.py
- tests/conftest.py

## Planning and Docs
- RAG_PIPELINE_AUDIT.md
- GOD_TIER_RAG_REBUILD.md
- .planning/ROADMAP.md
- .planning/phase-01-stabilize/PLAN.md
