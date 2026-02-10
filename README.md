# Multi-Agent RAG Pipeline

A sophisticated **Retrieval Augmented Generation (RAG)** pipeline with multiple AI agents for intelligent, factually-grounded question answering.

## Overview

This project implements a production-ready RAG system that combines retrieval, guardrails, generation, and evaluation into a cohesive pipeline. The system uses LLM-based agents at each stage to ensure answer quality, safety, and factual consistency.

### Key Features

- 🔍 **Vector-based Retrieval** – FAISS-backed search with similarity scores
- 🛡️ **Relevance & Safety Guardrails** – LLM-based filtering for relevance and PII/injection detection
- 🤖 **Grounded Answer Generation** – Context-aware responses with source citations
- ✅ **Factual Consistency Evaluation** – Automatic scoring of answer reliability
- 🚀 **Multi-LLM Support** – Works with OpenAI, Groq, or any OpenAI-compatible API
- 📊 **Detailed Metrics** – Processing time, consistency scores, claim-level analysis

## Architecture

```
Query → Retriever → Guardrail Agent → Generator → Evaluator → Result
         ↓            ↓                 ↓          ↓
      FAISS        LLM Filter      LLM Generate  LLM Score
      Search       Relevance       Grounded      Factual
                   & Safety        Answer        Consistency
```

### Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Retriever** | Fetch relevant chunks from knowledge base | FAISS + Sentence-Transformers |
| **Guardrail Agent** | Filter for relevance & safety (LLM #2) | OpenAI/Groq API |
| **Generator** | Produce grounded answers (LLM #3) | OpenAI/Groq API |
| **Evaluator** | Score factual consistency (LLM #4) | OpenAI/Groq API |
| **Vector Store** | Store embeddings with FAISS | FAISS |
| **Embedding Model** | Encode text to vectors | all-MiniLM-L6-v2 |
| **Document Loader** | Chunk documents for ingestion | Custom overlapping chunker |

## Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. **Clone or navigate to the project**
   ```bash
   cd rag_pipeline
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env  # or create a new .env file
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```dotenv
# LLM Configuration (OpenAI-compatible APIs)
OPENAI_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1  # or https://api.groq.com/openai/v1
LLM_MODEL=gpt-4o-mini  # or llama-3.3-70b-versatile for Groq
```

#### Supported LLM Providers

**OpenAI:**
```dotenv
OPENAI_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

**Groq (Free tier available):**
```dotenv
OPENAI_API_KEY=gsk_...  # Groq API key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### Configuration Classes (config.py)

All settings can be modified in `config.py`:

- **EmbeddingConfig** – Embedding model, dimension, device
- **VectorStoreConfig** – FAISS index type (flat/IVF/HNSW), persistence
- **RetrieverConfig** – Top-k, similarity threshold, MMR settings
- **ChunkingConfig** – Chunk size, overlap, minimum length
- **LLMConfig** – API key, base URL, timeouts, retry logic
- **GuardrailConfig** – Model, temperature, relevance threshold
- **GeneratorConfig** – Model, temperature, max tokens
- **EvaluatorConfig** – Model, temperature, consistency threshold

## Usage

### Basic Example

```python
from main import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
documents = [
    "Quantum computers use qubits instead of classical bits...",
    "Machine learning requires massive computational resources..."
]
pipeline.ingest(documents)

# Query the pipeline
result = pipeline.query("How do quantum computers achieve speedups?")

# Access results
print(result.answer)  # Generated answer text
print(result.consistency_score)  # 0.0–1.0 score
print(result.processing_time_ms)  # Total pipeline time
```

### Run Interactive Demo

```bash
python demo.py
```

The demo includes:
- Knowledge base ingestion (5 documents, 11 chunks)
- Pre-built queries with full pipeline visualization
- Interactive Q&A mode
- Detailed metrics and consistency breakdowns

### Ingest Documents from File

```python
from document_loader import DocumentLoader
from config import ChunkingConfig

loader = DocumentLoader(ChunkingConfig())
doc = loader.load_file("path/to/document.txt")
chunks = loader.chunk_documents([doc])
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Coverage

- **test_evaluator.py** – Guardrail agent filtering and safety detection
- **test_generator.py** – Generator output validation
- **test_guardrail.py** – Relevance filtering and safety flags
- **test_vector_store.py** – FAISS operations, persistence, MMR search

All 13 tests pass ✅

## Project Structure

```
rag_pipeline/
├── main.py                    # RAGPipeline orchestrator
├── config.py                  # Centralized configuration
├── models.py                  # Pydantic models for all data structures
├── llm_client.py              # OpenAI SDK wrapper with retry logic
├── embeddings.py              # Sentence-Transformers wrapper
├── vector_store.py            # FAISS-backed vector store
├── document_loader.py         # Document chunking & ingestion
├── retriever.py               # Query embedding & search
├── guardrail_agent.py         # Relevance & safety filtering (LLM #2)
├── generator.py               # Answer generation (LLM #3)
├── evaluator_agent.py         # Factual consistency scoring (LLM #4)
├── demo.py                    # Interactive demo with sample data
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create yourself)
├── README.md                  # This file
└── tests/
    ├── conftest.py            # Pytest configuration
    ├── test_evaluator.py      # Evaluator agent tests
    ├── test_generator.py      # Generator tests
    ├── test_guardrail.py      # Guardrail agent tests
    └── test_vector_store.py   # Vector store tests
```

## Key Configuration Options

### Retrieval (retriever.py)

```python
RetrieverConfig(
    top_k=10,                          # Initial retrieval count
    similarity_threshold=0.3,          # Min cosine similarity
    use_mmr=True,                      # Maximal Marginal Relevance
    mmr_lambda=0.7,                    # Diversity vs relevance trade-off
    mmr_top_k=5,                       # Final count after MMR
)
```

### Chunking (document_loader.py)

```python
ChunkingConfig(
    chunk_size=512,                    # Characters per chunk
    chunk_overlap=64,                  # Overlapping characters
    min_chunk_size=50,                 # Discard tiny tail chunks
)
```

### Vector Store (vector_store.py)

```python
VectorStoreConfig(
    index_type="flat",                 # "flat" | "ivf" | "hnsw"
    n_lists=100,                       # For IVF
    n_probe=10,                        # For IVF
    persist_dir="./vector_store_data", # Directory for persistence
)
```

## Performance Notes

- **Embedding Model**: all-MiniLM-L6-v2 (384-dim, ~25MB)
- **Vector Index**: FAISS flat index suitable for ~100k documents
- **Processing Time**: ~3-5 seconds per query (3-4 LLM calls)
- **Memory Usage**: Minimal; embeddings loaded on-demand

### Scaling Recommendations

- **100k+ documents**: Switch to IVF index (`index_type="ivf"`)
- **1M+ documents**: Use HNSW index (`index_type="hnsw"`)
- **Faster embeddings**: Use GPU-accelerated model with `device="cuda"`

## API Response Format

```python
class PipelineResult(BaseModel):
    query: str
    answer: str
    consistency_score: float           # 0.0–1.0 factual consistency
    is_reliable: bool                  # consistency_score >= threshold
    processing_time_ms: int
    retrieval_summary: RetrievalSummary
    guardrail_output: GuardrailOutput
    generator_output: GeneratorOutput
    evaluator_output: EvaluatorOutput
```

## Troubleshooting

### ModuleNotFoundError when running tests
- Ensure you're in the project root directory
- The `conftest.py` file automatically adds the parent directory to Python path

### API Key Authentication Error
- Verify `OPENAI_API_KEY` is correctly set in `.env`
- Check API key permissions and usage limits
- For Groq, ensure the key is from https://console.groq.com

### Out of Memory on Embeddings
- Reduce `batch_size` in `EmbeddingConfig`
- Process documents in smaller batches with `pipeline.ingest()`

### Slow Vector Search
- Use `index_type="ivf"` or `"hnsw"` for large databases
- Adjust `n_probe` (lower = faster but less accurate)

## Dependencies

```
openai>=1.30.0              # LLM client
faiss-cpu>=1.7.4            # Vector search (use faiss-gpu for GPU)
sentence-transformers>=2.7.0 # Embeddings
numpy>=1.24.0               # Numerical operations
pydantic>=2.5.0             # Data validation
tiktoken>=0.7.0             # Token counting
python-dotenv>=1.0.0        # Environment variables
rich>=13.7.0                # Terminal formatting
tenacity>=8.2.0             # Retry logic
pytest>=8.0.0               # Testing
```

## Future Enhancements

- [ ] Streaming response support for long answers
- [ ] Fine-tuning agents on domain-specific tasks
- [ ] Caching layer for frequent queries
- [ ] Multi-language support
- [ ] Batch processing API
- [ ] Web interface/API server
- [ ] Monitoring and analytics dashboard

## License

[Your License Here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues, please create a GitHub issue.

---

**Last Updated**: February 10, 2026
**Status**: IN PROGRESS