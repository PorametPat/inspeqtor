## List of tutorials for building this project

- [`uv`](https://docs.astral.sh/uv/concepts/projects/init/#applications) for package manager.
- [Github action](https://www.youtube.com/watch?v=Y6D2XaFV3Cc) for CI/CD.
- [Sphinx tutorial](https://youtu.be/KKfQnxQBoWE?si=JbCm1rioptxYjqMW) for documentation.

## Useful command lines

-  To create a new project using `uv` package manager, run the following command:

```bash
    uv init inspeqtor --lib
```

Using the `--lib` flag will create a library project. Remove the flag if you want to create an application project.

- To create virtual enviroment, run the following command:

```bash
    uv venv [name] --python 3.12
```
- The `[name]` is optional. If you don't provide a name, the virtual environment will be created with the name `venv`.
- If you use `[name]` you might have to use

```bash
    export UV_PROJECT_ENVIRONMENT=[name]
```
and also 
```bash
    export VIRTUAL_ENV=[name]
```
so that, `uv` will know which virtual environment to use.

- To install the `inspeqtor` package, run the following command:

```bash
    uv add . --editable --dev
```

To make the virtual environment discoverable in jupyter notebook, install the `ipykernel` package:

```bash
    uv add ipykernel --dev
```

- We use `pytest` for testing. To run the tests, run the following command:

```bash
    uv run pytest tests/ -v 
```
Flag `-v` is for verbose mode. The current module that is expected to be in used is `experimental` submodule. So the test command should be
```bash
uv run pytest tests/experimental/. -vv --durations=0
```
where the flag is for complehensive reporting.

- We use `ruff` for linting and formatting. To run the linter, run the following command:

```bash
    uvx ruff check .
```
For formatting, run the following command:

```bash
    uvx ruff format --check .
```

- We use `pyright` for type checking. To run the type checker, run the following command:

```bash
    uv run pyright .
```
For pre-commit run the following command:

```bash
    uv run pre-commit run --all-files
```

To perform profiling, we use `tuna`

1. Run the following command to the script you want to profile:

```bash
    uv run python -X importtime profile.py 2> import.log
```

2. Run the following command for visualization:

```bash
    uvx tuna import.log
```

## For using sphinx auto genereate documentation:

Setting things up
```bash
$ documents/
uv run sphinx-quickstart
```

Read the src files
```bash
$ root
uv run sphinx-apidoc -o documents/ src/inspeqtor/
```

Make the HTML files
```bash
$ documents/
uv run make html
```

We use `furo` for the theme of sphinx
```bash
uv add furo --dev
```

