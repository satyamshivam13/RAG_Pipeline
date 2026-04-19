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
| `python demo.py` | Run interactive end-to-end demo. |
| `pytest tests/ -v` | Run full automated tests. |

## Code Style

No dedicated linter/formatter config files are present (no `ruff`, `black`, `flake8`, or `pylintrc` config discovered in repository root).

Recommended local checks:

- Keep imports organized and avoid unused code paths.
- Use descriptive typed pydantic/dataclass models as established in `models.py` and `config.py`.
- Run tests before committing.

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
