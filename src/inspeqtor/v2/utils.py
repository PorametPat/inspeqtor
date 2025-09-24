import jax
import jax.numpy as jnp
import typing
import logging
from dataclasses import dataclass

from inspeqtor.experimental.data import QubitInformation
from inspeqtor.v2.control import ControlSequence
from inspeqtor.v2.data import ExperimentalData


@dataclass
class LoadedData:
    experiment_data: ExperimentalData
    control_parameters: jnp.ndarray
    unitaries: jnp.ndarray
    observed_values: jnp.ndarray
    control_sequence: ControlSequence
    whitebox: typing.Callable
    noisy_whitebox: typing.Callable | None = None
    noisy_unitaries: jnp.ndarray | None = None


@dataclass
class SyntheticDataModel:
    control_sequence: ControlSequence
    qubit_information: QubitInformation
    dt: float
    ideal_hamiltonian: typing.Callable[..., jnp.ndarray]
    total_hamiltonian: typing.Callable[..., jnp.ndarray]
    solver: typing.Callable[..., jnp.ndarray]
    quantum_device: typing.Callable[..., jnp.ndarray] | None
    whitebox: typing.Callable[..., jnp.ndarray] | None


def prepare_data(
    exp_data: ExperimentalData,
    control_sequence: ControlSequence,
    whitebox: typing.Callable,
) -> LoadedData:
    """Prepare the data for easy accessing from experiment data, control sequence, and Whitebox.

    Args:
        exp_data (ExperimentData): `ExperimentData` instance
        control_sequence (ControlSequence): Control sequence of the experiment
        whitebox (typing.Callable): Ideal unitary solver.

    Returns:
        LoadedData: `LoadedData` instance
    """
    logging.info(f"Loaded data from {exp_data.config.EXPERIMENT_IDENTIFIER}")

    control_parameters = exp_data.get_parameter()

    expectation_values = exp_data.get_observed()
    unitaries = jax.vmap(whitebox)(control_parameters)

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.config.EXPERIMENT_IDENTIFIER}"
    )

    return LoadedData(
        experiment_data=exp_data,
        control_parameters=control_parameters,
        unitaries=unitaries[:, -1, :, :],
        observed_values=expectation_values,
        control_sequence=control_sequence,
        whitebox=whitebox,
    )
