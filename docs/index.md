---

title: One Solid Step at a Time C&C for Quantum Device

description: A Data-Efficient Framework for Characterizing and Calibrating That Squeezes Every Bit of Information from Quantum Devices.

---

<div style="text-align: center;">

    <img src="assets/inspeqtor_logo.svg" alt="Alt Text" style="width:15%; height:auto;">

    <p style="font-weight: bold; font-family: monospace;">

        inspeqtor

    </p>

</div>

# Greeting 🖖

We aim to provide a data-efficient framework for the characterization and calibration of quantum devices. Since running experiments and obtaining data from quantum devices can be expensive, we strive to extract every bit of information from the data collected.

!!! warning

    This package and its documentation are still under development. Please proceed with caution.

## Installation

To install `inspeqtor` from the PyPI:

=== "uv"

    ```bash

    uv add inspeqtor

    ```

=== "pip"

    ```bash

    pip install inspeqtor

    ```

To install `inspeqtor` from the remote repository:

=== "uv"

    ```bash

    uv add git+https://github.com/PorametPat/inspeqtor.git

    ```

=== "pip"

    ```bash

    pip install git+https://github.com/PorametPat/inspeqtor.git

    ```

To install `inspeqtor` locally for development, clone the repository:

```bash
git clone https://github.com/PorametPat/inspeqtor.git
```

Then, enter the cloned directory

```bash
cd inspeqtor
```

and install it using:

=== "uv"

    ```bash

    uv sync

    ```

=== "pip"

    ```bash

    pip install ./<PATH>

    ```

where `<PATH>` is the path to your local `inspeqtor` repository.

## What you can do with `inspeqtor`

Currently, the library fully supports:

- Characterizing single-qubit devices using the Graybox characterization method.

- Open-loop and closed-loop control optimization with gradient-based optimization and Bayesian Optimization.

- For characterization, users can choose between a `statistical` or `probabilistic` model.

- With the `probabilistic` model, users can leverage `boed` to perform Bayesian Optimal Experiment Design for data-efficient characterization.

## The Road Ahead

The API is evolving to fully support general subsystem characterization methods. Future features will include:

- Defining controls for selected characterization methods.

- Performing experiments and storing data using our unified local file system approach.

- Loading experimental data into memory and characterizing model parameters with our optimizers.

## Next Step

[Overviews](./tutorials/overviews.md) is a good place to start.

## Citation

If you find our library to be useful and would like to cite our work, please use the following,

```bibtex
@software{Pathumsoot_inspeqtor,
    author = {Pathumsoot, Poramet},
    doi = {10.5281/zenodo.17748402},
    license = {BSD-3-Clause},
    title = {{inspeqtor}},
    url = {https://github.com/PorametPat/inspeqtor}
}
```
