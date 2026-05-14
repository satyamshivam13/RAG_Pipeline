<!-- generated-by: gsd-doc-writer -->
# Development

## Local Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

For local CI parity, install developer tooling too:

```bash
pip install -r requirements-dev.txt
```

3. Copy sample environment variables.

```bash
copy .env.example .env
```

4. Run a quick manual check.

```bash
python demo.py
```

## Build Commands

| Command | Description |
|---|---|
| `pip install -r requirements.txt` | Install runtime and test dependencies. |
| `pip install -r requirements-dev.txt` | Install runtime dependencies plus CI tooling. |
| `python demo.py` | Run interactive end-to-end demo. |
| `pytest tests/ -v` | Run full automated tests. |
| `black --check --diff .` | Verify formatting. |
| `flake8 .` | Run lint checks. |
| `mypy` | Run static type checks using `pyproject.toml`. |
| `bandit -r . -c pyproject.toml` | Run source security scan. |
| `safety check --file requirements.txt --full-report` | Run dependency vulnerability scan. |

## Code Style

Formatting, linting, and typing are configured in `pyproject.toml` and `.flake8`.

Recommended local checks:

- Keep imports organized and avoid unused code paths.
- Use descriptive typed pydantic/dataclass models as established in `models.py` and `config.py`.
- Run tests before committing.

## CI/CD

GitHub Actions are split into a fast PR lane and an optional expensive evaluation lane.

- `CI` runs on pull requests and pushes to `main`: Black, flake8, mypy, pytest, Bandit, and Safety.
- `Evaluation` runs weekly, manually, or when a PR has the `run-evaluation` label. This preserves quality regression signal without slowing every PR.
- Workflow summaries and uploaded artifacts provide failure context for pytest, Bandit, and evaluation reports.
- Python dependency caching is enabled through `actions/setup-python` using `requirements.txt`, `requirements-dev.txt`, and tool config files as cache keys.

## Branch Conventions

No branch naming convention is documented in this repository.

A practical convention for this codebase:

- `feat/<short-topic>` for new features.
- `fix/<short-topic>` for bug fixes.
- `docs/<short-topic>` for documentation-only changes.

## PR Process

No pull request template is present in `.github/`.

Suggested process for contributors:

1. Keep changes focused and scoped to one concern.
2. Run `pytest tests/ -v` and include the result in the PR description.
3. Describe behavioral impact (runtime path, config changes, or model changes).
4. Include before/after notes for pipeline output behavior when relevant.
