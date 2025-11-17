# Stable API Documentation

This directory contains the complete API reference for the stable `inspeqtor` namespace.

## Structure

All modules are documented using the stable import path `inspeqtor.*` rather than the underlying implementation paths (`inspeqtor.v2.*` or `inspeqtor.experimental.*`).

### Modules

- **[data.md](./data.md)** - Data structures and I/O operations
  - `QubitInformation`, `DataBundled`, `ExperimentalData`, `ExperimentConfiguration`
  - Data loading/saving functions
  - Data preparation utilities
  - **library** submodule: Predefined data generators and mock utilities

- **[control.md](./control.md)** - Control sequence definitions and operations
  - `BaseControl`, `ControlSequence`
  - Control waveform generation and manipulation
  - Parameter sampling and transformation
  - **library** submodule: Pre-built pulse shapes (DRAG, Gaussian), feature maps

- **[models.md](./models.md)** - Predictive models and neural network utilities
  - **adapter** submodule: Observable and unitary adapters
  - **shared** submodule: Common model utilities (MSE, constraints, training)
  - **library** submodule: Pre-built models (overview)
    - **[linen](./models-library-linen.md)**: Flax Linen models (`WoModel`, `UnitaryModel`, etc.)
    - **[nnx](./models-library-nnx.md)**: Flax NNX models (`WoModel`, `UnitaryModel`, etc.)
  - **probabilistic** submodule: Bayesian neural network support

- **[optimize.md](./optimize.md)** - Optimization algorithms
  - Gradient-based optimization (`minimize`, `stochastic_minimize`)
  - Gaussian process regression
  - Bayesian optimization utilities (`BayesOptState`, expected improvement)

- **[physics.md](./physics.md)** - Quantum physics simulation
  - Hamiltonian operators and transformations
  - Solvers (including Lindblad and Trotterization)
  - Fidelity calculations and state tomography
  - **library** submodule: Predefined Hamiltonians (transmon, rotating frame, etc.)

- **[utils.md](./utils.md)** - Utility functions
  - Synthetic data generation
  - JAX utilities and data transformations
  - Visualization functions
  - Constants (Pauli operators, expectation value order)

- **[boed.md](./boed.md)** - Bayesian Optimal Experiment Design
  - Loss functions for EIG estimation (marginal, VNMC)
  - Guide initialization and optimization
  - Vectorization utilities

## Documentation Format

Each module file follows this pattern:

```markdown
# Module Name

::: inspeqtor.module_name

::: inspeqtor.module_name.ClassName

::: inspeqtor.module_name.function_name

...
```

This ensures:

- Users see the stable API path in documentation
- Each class/function gets its own section
- Documentation is auto-generated from docstrings
- Cross-references work correctly

## Maintenance

When adding new exports to `src/inspeqtor/stable/<module>/__init__.py`, remember to:

1. Add a corresponding `::: inspeqtor.<module>.<symbol>` entry to the relevant `.md` file
2. Keep the order logical (classes first, then functions, or alphabetical)
3. Test with `uv run mkdocs build --strict`
