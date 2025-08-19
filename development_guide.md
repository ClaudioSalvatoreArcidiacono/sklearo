# Development Guide

## Installing dev dependencies

```bash
# Install uv (one time)
curl -fsSL https://astral.sh/uv/install.sh | sh

# Install dependencies (creates .venv in the project)
uv sync --extra dev

# Run tests
uv run pytest
```
