import jax
import jax.numpy as jnp
import typing
import logging
from dataclasses import dataclass

from inspeqtor.experimental.data import QubitInformation
from inspeqtor.experimental.physics import calculate_exp
from inspeqtor.v2.control import ControlSequence
from inspeqtor.v2.data import (
    ExpectationValue,
    ExperimentalData,
    get_complete_expectation_values,
    get_initial_state,
    get_observable_operator,
)
from inspeqtor.experimental.utils import calculate_shots_expectation_value


@dataclass
class LoadedData:
    """A utility dataclass holding objects necessary for device characterization."""

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
    """A utility dataclass holding objects necessary for simulating single qubit quantum device."""

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


def calculate_expectation_values(
    unitaries: jnp.ndarray,
    expectation_value_order: list[ExpectationValue] = get_complete_expectation_values(
        1
    ),
) -> jnp.ndarray:
    # Calculate the ideal expectation values of the original pulse
    ideal_expectation_values = jnp.zeros(tuple(unitaries.shape[:-2]) + (18,))
    for idx, exp in enumerate(expectation_value_order):
        expvals = calculate_exp(
            unitaries,
            get_observable_operator(exp.observable),
            get_initial_state(exp.initial_state, dm=True),
        )
        ideal_expectation_values = ideal_expectation_values.at[..., idx].set(expvals)

    return ideal_expectation_values


def shot_quantum_device(
    key: jnp.ndarray,
    control_parameters: jnp.ndarray,
    solver: typing.Callable[[jnp.ndarray], jnp.ndarray],
    SHOTS: int,
    expectation_value_receipt: typing.Sequence[
        ExpectationValue
    ] = get_complete_expectation_values(1),
) -> jnp.ndarray:
    """This is the shot estimate expectation value quantum device

    Args:
        control_parameters (jnp.ndarray): The control parameter to be feed to simlulator
        key (jnp.ndarray): Random key
        solver (typing.Callable[[jnp.ndarray], jnp.ndarray]): The ODE solver for propagator
        SHOTS (int): The number of shots used to estimate expectation values

    Returns:
        jnp.ndarray: The expectation value of shape (control_parameters.shape[0], 18)
    """

    expectation_values = jnp.zeros(
        (control_parameters.shape[0], len(expectation_value_receipt))
    )
    unitaries = jax.vmap(solver)(control_parameters)[:, -1, :, :]

    for idx, exp in enumerate(expectation_value_receipt):
        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, num=unitaries.shape[0])

        expectation_value = jax.vmap(
            calculate_shots_expectation_value,
            in_axes=(0, None, 0, None, None),
        )(
            sample_keys,
            get_initial_state(exp.initial_state, dm=True),
            unitaries,
            get_observable_operator(exp.observable),
            SHOTS,
        )

        expectation_values = expectation_values.at[..., idx].set(expectation_value)

    return expectation_values


def dictorization(expvals: jnp.ndarray, order: list[ExpectationValue]):
    """This function formats expectation values of shape (18, N) to a dictionary
    with the initial state as outer key and the observable as inner key.

    Args:
        expvals (jnp.ndarray): Expectation values of shape (18, N). Assumes that order is as in default_expectation_values_order.

    Returns:
        dict[str, dict[str, jnp.ndarray]]: A dictionary with the initial state as outer key and the observable as inner key.
    """
    expvals_dict: dict[str, dict[str, jnp.ndarray]] = {}
    for idx, exp in enumerate(order):
        if exp.initial_state not in expvals_dict:
            expvals_dict[exp.initial_state] = {}

        expvals_dict[exp.initial_state][exp.observable] = expvals[idx]

    return expvals_dict
