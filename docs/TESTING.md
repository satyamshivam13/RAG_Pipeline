<!-- generated-by: gsd-doc-writer -->
# Testing

## Test Framework and Setup

- Framework: `pytest` (declared in `requirements.txt` as `pytest>=8.0.0`).
- Setup helper: `tests/conftest.py` prepends the project root to `sys.path` for module imports.

Before running tests:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Tests

Run full suite:

```bash
pytest tests/ -v
```

Run a single test module:

```bash
pytest tests/test_vector_store.py -v
```

Run a single test case by name:

```bash
pytest tests/test_pipeline_phase1_runtime.py -k sync_mode -v
```

## Writing New Tests

Current test naming pattern:

- Directory: `tests/`
- Files: `test_*.py`
- Test functions: `test_*`

Useful patterns already in repo:

- Monkeypatch-based runtime isolation (`tests/test_pipeline_phase1_runtime.py`) for deterministic unit testing.
- Direct component tests for loader, embeddings, retriever/generator/evaluator, and vector store.

## Coverage Requirements

No explicit coverage threshold config was found (no `.coveragerc`, `pytest-cov` config, or CI-enforced threshold).

## CI Integration

No CI workflow files were found in `.github/workflows/`.

Tests currently run as a local developer responsibility via `pytest` commands.
