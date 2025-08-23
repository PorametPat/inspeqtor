# Dataset Creating

## Prepare experiments

We break the experiment preparation phase into the following steps.

- Gathering "prior" information about the quantum device.
- Defining the control action.

### Quantum device specification

In `inspeqtor`, we focus on characterizing quantum device. In the finest level, user most likely want to perform control calibration on individual qubit which is part of the full system. Thus, we provide a dedicated `dataclass` responsible for holding "prior" information about the qubit. The information is often necessary for constructing the subsystem Hamiltonian which is used for open-loop optimization. Below is the code snippet to initialize `QubitInformation` object.

```python
--8<-- "guides/gen_data.py:qubit-info"
```

### Define the control

For composability of control action, we let user define an "atomic" control action by inheriting `Control` dataclass, then compose them together via `ControlSequence` class. Below is an example of defining the total control action with only single predefined `DragPulseV2`.

```python
--8<-- "guides/gen_data.py:control"
```

## Perform experiment

### Generate some synthetic dataset to work with

```python
--8<-- "guides/gen_data.py:gen-syn-dataset"
```

### Perform experiment on actual device

## Save the experiment

```python
--8<-- "guides/gen_data.py:save-dataset"
```

## Load the experiment

```python
--8<-- "guides/gen_data.py:load-dataset"
```