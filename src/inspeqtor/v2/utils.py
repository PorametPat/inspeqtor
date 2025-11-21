import jax
import jax.numpy as jnp
import typing
import logging
from dataclasses import dataclass
from itertools import product

from inspeqtor.v1.data import QubitInformation
from inspeqtor.v1.physics import calculate_exp
from inspeqtor.v2.control import ControlSequence
from inspeqtor.v2.data import (
    ExpectationValue,
    ExperimentalData,
    get_complete_expectation_values,
    get_initial_state,
    get_observable_operator,
    check_parity,
)
from inspeqtor.v1.utils import calculate_shots_expectation_value
from inspeqtor.v2.constant import plus_projectors, minus_projectors


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


def single_qubit_shot_quantum_device(
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


def count_bits(
    binaries: jnp.ndarray, n_qubits: int, axis: int = 1
) -> dict[str, jnp.ndarray]:
    return {
        str(integer): jnp.sum(binaries == integer, axis=axis)
        for integer in range(int(2**n_qubits))
    }


Param = typing.TypeVar("Param")


def tensor_product(*operators) -> jnp.ndarray:
    """Create tensor product of multiple operators"""
    return jax.tree.reduce(jnp.kron, operators)


def finite_shot_expectation_value(key: jnp.ndarray, prob: jnp.ndarray, shots: int):
    return jnp.mean(
        jax.random.choice(
            key,
            jax.vmap(check_parity)(jnp.arange(0, prob.size, dtype=jnp.int_)),
            shape=(shots,),
            p=prob,
        )
    )


def finite_shot_quantum_device(
    key: jnp.ndarray,
    param: Param,
    solver: typing.Callable[[Param, jnp.ndarray], jnp.ndarray],
    shots: int,
    expval: ExpectationValue,
):
    initial_state = get_initial_state(expval.initial_state, dm=True)

    state = solver(param, initial_state)
    prob = get_measurement_probability(state, expval.observable)

    return finite_shot_expectation_value(key, prob, shots)


projectors = {
    "X": {
        0: plus_projectors["X"],
        1: minus_projectors["X"],
    },
    "Y": {
        0: plus_projectors["Y"],
        1: minus_projectors["Y"],
    },
    "Z": {
        0: plus_projectors["Z"],
        1: minus_projectors["Z"],
    },
}


def get_measurement_probability(state: jnp.ndarray, operator: str) -> jnp.ndarray:
    """Calculate the probability of measuring each projector of tensor product of Pauli operators

    Args:
        state (jnp.ndarray): The quantum state to measure
        operator (str): The string representation of the measurement operator, e.g., 'XY'

    Returns:
        jnp.ndarray: An array of probability where each index is a base 10 representation of base 2 measurement result.
    """

    return jnp.array(
        [
            jnp.trace(state @ tensor_product(*g_projector))
            for g_projector in product(
                *[(projectors[op][0], projectors[op][1]) for op in operator]
            )
        ]
    ).real
