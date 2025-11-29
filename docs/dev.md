# Developer Guide for Inspeqtor

## Getting Started

This guide provides the necessary steps and commands for developing the Inspeqtor package. We use the `uv` package manager for dependency management and various tools for testing, linting, and documentation.

## How to contribute?

First of all, thank you for interested in contributing to `inspeqtor` 😊

By contributing your work to this repository, you agree to license it in perpetuity under the terms described in [LICENSE.md](https://github.com/PorametPat/inspeqtor/blob/main/LICENSE.md). You are also asserting that the work is your own, and that you have the right to license it to us.

If you wish to integrate our work into your own projects, please follow the attribution guidelines in [LICENSE.md](https://github.com/PorametPat/inspeqtor/blob/main/LICENSE.md).

We use git for our library developement. The following steps will guide you to setup everthing necessary for developement. If you encount any problem, please do not hesitate to reach out to us by creating github issue at our github repository.

Clone the project from remote to your local machine and `cd` into the directory.

```bash
git clone https://github.com/PorametPat/inspeqtor.git && cd inspeqtor
```

We recommend using `uv` for environment and dependencies manangement. To learn more about `uv`, please see their [official documentation](https://docs.astral.sh/uv/). You can create the virtual environment along with installing dependencies using the following commands. Use the appropiate command corresponding to what you want to do, .e.g, `docs` for documentation contribution.

=== "Minimal"

    ```bash title="Minimal requirement"
    uv sync 
    ```

=== "dev"

    ```bash title="With developement dependencies"
    uv sync --group dev
    ```

=== "docs"

    ```bash title="With documentation dependencies"
    uv sync --group docs
    ```

After modification, please use the following command to check the code quality to make the development life cycle faster. Note that we also use workflow to run the automate test, but it take a longer time than local testing.

```bash title="Using pre-commit hooks"
uv run pre-commit run --all-files
```

Or using script, we may need to make the script executable first by using

```bash
chmod +x ./bin/check.bash
```

then execute it using,

```bash title="Using script"
./bin/check.bash
```

Please read the following for more details.

## Project Setup

The following is the instruction used for project initialization.

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

```bash title="Intall dependency in optional docs group"
uv add mkdocs-marimo --optional docs
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
uv run pytest tests/. -vv --durations=0
```

```bash title="Test with specific Python version"
uv run --python 3.12 --with '.[test]' pytest tests/.
```

```bash title="Test docstrings"
uv run -m doctest src/inspeqtor/v1/utils.py
```

```bash title="Test with live logging"
uv run -m doctest src/inspeqtor/v1/utils.py --log-cli-level=INFO
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

If an error `"cairosvg" Python module is installed, but it crashed with:` occur, please check [this link](https://squidfunk.github.io/mkdocs-material/plugins/requirements/image-processing/#troubleshooting) for more details of how to solve it. Otherwise, if you are using macOS and the issue persist after more than one resolved, the following command line might be useful:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

After execution of the above command, please run `serve` again.

The following command will deploy the document to the github page. However, since we are using `mike` for documentaion versioning please see command in [Versioning](#versioning) for deployment instead. 

```bash title="Deploy to GitHub Pages using mkdocs"
mkdocs gh-deploy
```

### Versioning

!!! note
    The versioning process should be automated. But we document the process here for explanation.

We use mike to manage documentation versioning. See the following references for more details:

- [Example Git repo](https://github.com/mkdocs-material/example-versioning)
- [Official Mkdocs about versioning](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)

Here are the commands related to the versioning.

To list all of the versions of the documentation use

```bash
uv run mike list
```

After making change, and you want to change the documentation's version, use the following command to update the version number and set the aliases to `latest`. In the below snippet, we use version of `0.2`,

```bash
uv run mike deploy --push --update-aliases 0.2 latest
```

To make the defualt version of the documentation to `latest` use

```bash
uv run mike set-default --push latest
```

If you want to simply update the documentation with the current version (e.g. `0.1`) use,

```bash
uv run mike deploy --push 0.1
```

Note that the `--push` option will push the change to `gh-pages` branch directly. Furthermore, mike will deploy the documentation built from the latest branch of the main branch.

## Performance Profiling

```bash title="Generate import profile"
uv run python -X importtime profile.py > import.log
```

```bash title="Visualize import profile"
uvx tuna import.log
```

## CI/CD

For setting up GitHub Actions for CI/CD, refer to this [tutorial](https://www.youtube.com/watch?v=Y6D2XaFV3Cc).

---

This guide covers the essential commands and workflows for developing the Inspeqtor package. For more detailed information about specific components, refer to the respective documentation.
