# Developer Guide for Inspeqtor

## Getting Started

This guide provides the necessary steps and commands for developing the Inspeqtor package. We use the `uv` package manager for dependency management and various tools for testing, linting, and documentation.

## Project Setup

### Creating a New Project

```bash title="Create a library project"
uv init inspeqtor --lib
```

```bash title="Create an application project"
uv init inspeqtor
```

### Setting Up a Virtual Environment

```bash title="Create a virtual environment with Python 3.12"
uv venv [name] --python 3.12
```

If using a custom name for your virtual environment, set these environment variables:

```bash title="Set environment variable for custom venv name"
export UV_PROJECT_ENVIRONMENT=[name]
```

```bash title="Set VIRTUAL_ENV environment variable"
export VIRTUAL_ENV=[name]
```

### Installing the Package

```bash title="Install the package in development mode"
uv add . --editable --dev
```

```bash title="Install Jupyter integration"
uv add ipykernel --dev
```

### Syncing dependencies

```bash title="Install all optional dependencies"
uv sync --all-extras
```

## Development Workflow

The project uses several tools for development:

- **pytest** for testing
- **ruff** for linting and formatting
- **pyright** for type checking
- **pre-commit** for git hooks
- **tuna** for import profiling

## Testing

```bash title="Basic test run"
uv run pytest tests/ -v
```

```bash title="Test experimental module with detailed output"
uv run pytest tests/experimental/. -vv --durations=0
```

```bash title="Test with specific Python version"
uv run --python 3.12 --with '.[test]' pytest tests/experimental/.
```

```bash title="Test docstrings"
uv run -m doctest src/inspeqtor/experimental/utils.py
```

## Code Quality

```bash title="Run linting with Ruff"
uvx ruff check .
```

```bash title="Check code formatting"
uvx ruff format --check .
```

```bash title="Run type checking with Pyright"
uv run pyright .
```

```bash title="Run pre-commit hooks"
uv run pre-commit run --all-files
```

## Documentation

### MkDocs Setup

We use MkDocs Material and pymdown-extensions for documentation generation. Follow the [Real Python tutorial](https://realpython.com/python-project-documentation-with-mkdocs/#step-4-prepare-your-documentation-with-mkdocs) for detailed setup instructions.

```bash title="Serve documentation locally"
uv run mkdocs serve
```

```bash title="Deploy to GitHub Pages"
mkdocs gh-deploy
```

## Performance Profiling

```bash title="Generate import profile"
uv run python -X importtime profile.py 2> import.log
```

```bash title="Visualize import profile"
uvx tuna import.log
```

## CI/CD

For setting up GitHub Actions for CI/CD, refer to this [tutorial](https://www.youtube.com/watch?v=Y6D2XaFV3Cc).

---

This guide covers the essential commands and workflows for developing the Inspeqtor package. For more detailed information about specific components, refer to the respective documentation.
