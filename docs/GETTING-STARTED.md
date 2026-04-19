<!-- generated-by: gsd-doc-writer -->
# Getting Started

## Prerequisites

- Python `>=3.10`
- `pip`
- Network access to your configured OpenAI-compatible LLM endpoint

## Installation

1. Clone the repository.

```bash
git clone <your-repo-url>
```

2. Move into the project directory.

```bash
cd RAG_Pipeline-1
```

3. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

5. Create your local environment file.

```bash
copy .env.example .env
```

## First Run

Run the demo pipeline:

```bash
python demo.py
```

You should see retrieval, generation, and evaluation output in an interactive console workflow.

## Common Setup Issues

1. `Embedding dimension mismatch` during startup.
Cause: `EmbeddingConfig.dimension` does not match your chosen sentence-transformers model output.
Fix: align `EmbeddingConfig.dimension` with model output or switch model/dimension pair.

2. LLM authentication errors.
Cause: missing or invalid `OPENAI_API_KEY` for your selected endpoint.
Fix: set `OPENAI_API_KEY` in `.env`, verify `LLM_BASE_URL` and `LLM_MODEL` match provider expectations.

3. FAISS install/import problems.
Cause: incompatible Python environment or missing dependency install.
Fix: recreate `.venv`, then run `pip install -r requirements.txt` again.

## Next Steps

- Continue with development workflows in `docs/DEVELOPMENT.md`.
- Run and extend tests using `docs/TESTING.md`.
- Review architecture details in `docs/ARCHITECTURE.md`.
