import jax
import jax.numpy as jnp
import typing
from dataclasses import dataclass
from inspeqtor.experimental.control import ControlSequence
from inspeqtor.experimental.data import (
    ExpectationValue,
    ExperimentData,
    QubitInformation,
    State,
)
from flax.typing import VariableDict
from inspeqtor.experimental.model import mse
from inspeqtor.experimental.constant import (
    X,
    Y,
    Z,
    default_expectation_values_order,
    plus_projectors,
    minus_projectors,
    get_default_expectation_values_order,
)
from inspeqtor.experimental.decorator import warn_not_tested_function
from inspeqtor.experimental.physics import calculate_exp, HamiltonianArgs
import logging


@dataclass
class LoadedData:
    experiment_data: ExperimentData
    control_parameters: jnp.ndarray
    unitaries: jnp.ndarray
    observed_values: jnp.ndarray
    control_sequence: ControlSequence
    whitebox: typing.Callable
    noisy_whitebox: typing.Callable | None = None
    noisy_unitaries: jnp.ndarray | None = None


def center_location(num_of_pulse: int, total_time_dt: int | float) -> jnp.ndarray:
    """Create an array of location equally that centered each pulse.

    Args:
        num_of_pulse (int): The number of the pulse in the sequence to be equally center.
        total_time_dt (int | float): The total bins of the sequence.

    Returns:
        jnp.ndarray: The array of location equally that centered each pulse.
    """
    center_locations = (
        jnp.array([(k - 0.5) / num_of_pulse for k in range(1, num_of_pulse + 1)])
        * total_time_dt
    )
    return center_locations


def drag_envelope_v2(
    amp: float | jnp.ndarray,
    sigma: float | jnp.ndarray,
    beta: float | jnp.ndarray,
    center: float | jnp.ndarray,
    final_amp: float | jnp.ndarray = 1.0,
):
    """Drag pulse following: https://docs.quantum.ibm.com/api/qiskit/qiskit.pulse.library.Drag_class.rst#drag

    Args:
        amp (float | jnp.ndarray): The amplitude of the pulse
        sigma (float | jnp.ndarray): The standard deviation of the pulse
        beta (float | jnp.ndarray): DRAG coefficient.
        center (float | jnp.ndarray): Center location of the pulse
        final_amp (float | jnp.ndarray, optional): Final amplitude of the control. Defaults to 1.0.

    Returns:
        typing.Callable: DRAG envelope function
    """

    def g(t):
        return jnp.exp(-((t - center) ** 2) / (2 * sigma**2))

    def g_prime(t):
        return amp * (g(t) - g(-1)) / (1 - g(-1))

    def envelop(t):
        return final_amp * g_prime(t) * (1 + 1j * beta * (t - center) / sigma**2)

    return envelop


@warn_not_tested_function
def detune_hamiltonian(
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    detune: float,
) -> typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
    """Detune the Hamiltonian in Z-axis with detuning coefficient

    Args:
        hamiltonian (typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray]): Hamiltonian function to be detuned
        detune (float): Detuning coefficient

    Returns:
        typing.Callable: Detuned Hamiltonian.

    """

    def detuned_hamiltonian(
        params: HamiltonianArgs,
        t: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return hamiltonian(params, t, *args, **kwargs) + detune * Z

    return detuned_hamiltonian


def prepare_data(
    exp_data: ExperimentData,
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
    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    control_parameters = jnp.array(exp_data.parameters)
    # * Attempt to reshape the control_parameters to (size, features)
    if len(control_parameters.shape) == 3:
        control_parameters = control_parameters.reshape(
            control_parameters.shape[0],
            control_parameters.shape[1] * control_parameters.shape[2],
        )

    expectation_values = jnp.array(exp_data.get_expectation_values())
    unitaries = jax.vmap(whitebox)(control_parameters)

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    return LoadedData(
        experiment_data=exp_data,
        control_parameters=control_parameters,
        unitaries=unitaries[:, -1, :, :],
        observed_values=expectation_values,
        control_sequence=control_sequence,
        whitebox=whitebox,
    )


def random_split(key: jnp.ndarray, test_size: int, *data_arrays: jnp.ndarray):
    """The random_split function splits the data into training and testing sets.

    Examples:
        >>> key = jax.random.key(0)
        >>> x = jnp.arange(10)
        >>> y = jnp.arange(10)
        >>> x_train, y_train, x_test, y_test = random_split(key, 2, x, y)
        >>> assert x_train.shape[0] == 8 and y_train.shape[0] == 8
        >>> assert x_test.shape[0] == 2 and y_test.shape[0] == 2

    Args:
        key (jnp.ndarray): Random key.
        test_size (int): The size of the test set. Must be less than the size of the data.

    Returns:
        typing.Sequence[jnp.ndarray]: The training and testing sets in the same order.
    """
    # * General random split
    idx = jax.random.permutation(key, data_arrays[0].shape[0])
    train_data = []
    test_data = []

    for data in data_arrays:
        train_data.append(data[idx][test_size:])
        test_data.append(data[idx][:test_size])

    return (*train_data, *test_data)


def dataloader(
    arrays: typing.Sequence[jnp.ndarray],
    batch_size: int,
    num_epochs: int,
    *,
    key: jnp.ndarray,
):
    """The dataloader function creates a generator that yields batches of data.

    Args:
        arrays (typing.Sequence[jnp.ndarray]): The list or tuple of arrays to be batched.
        batch_size (int): The size of the batch.
        num_epochs (int): The number of epochs. If set to -1, the generator will run indefinitely.
        key (jnp.ndarray): The random key.

    Returns:
        None: stop the generator.

    Yields:
        typing.Any: (step, batch_idx, is_last_batch, epoch_idx), (array_batch, ...)
    """
    # * General dataloader
    # Check that all arrays have the same size in the first dimension
    dataset_size = arrays[0].shape[0]
    # assert all(array.shape[0] == dataset_size for array in arrays)
    # Generate random indices
    indices = jnp.arange(dataset_size)
    step = 0
    epoch_idx = 0
    while True:
        if epoch_idx == num_epochs:
            return None
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        batch_idx = 0
        start = 0
        end = batch_size
        is_last_batch = False
        while not is_last_batch:
            batch_perm = perm[start:end]
            # Check if this is the last batch
            is_last_batch = end >= dataset_size
            yield (
                (step, batch_idx, is_last_batch, epoch_idx),
                tuple(array[batch_perm] for array in arrays),
            )
            start = end
            end = start + batch_size
            step += 1
            batch_idx += 1

        epoch_idx += 1


def variance_of_observable(expval: jnp.ndarray, shots: int = 1):
    return (1 - expval**2) / shots


# dataset metrics
@dataclass
class DatasetMetrics:
    # The data variance
    var: float
    # The MSE between ideal and experimental expectation values
    mse_ideal2exp: float
    # The training iteration
    total_iterations: int
    step_for_optimizer: int
    warmup_steps: int
    cool_down_steps: int


def calculate_expectation_values(
    unitaries: jnp.ndarray,
) -> jnp.ndarray:
    # Calculate the ideal expectation values of the original pulse
    ideal_expectation_values = jnp.zeros(tuple(unitaries.shape[:-2]) + (18,))
    for idx, exp in enumerate(default_expectation_values_order):
        expvals = calculate_exp(
            unitaries,
            exp.observable_matrix,
            exp.initial_density_matrix,
        )
        ideal_expectation_values = ideal_expectation_values.at[..., idx].set(expvals)

    return ideal_expectation_values


def expectation_value_to_prob_plus(expectation_value: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the probability of -1 and 1 for the given expectation value
    E[O] = -1 * P[O = -1] + 1 * P[O = 1], where P[O = -1] + P[O = 1] = 1
    Thus, E[O] = -1 * (1 - P[O = 1]) + 1 * P[O = 1]
    E[O] = 2 * P[O = 1] - 1 -> P[O = 1] = (E[O] + 1) / 2
    Args:
        expectation_value (jnp.ndarray): Expectation value of quantum observable

    Returns:
        jnp.ndarray: Probability of measuring plus eigenvector
    """

    return (expectation_value + 1) / 2


def expectation_value_to_prob_minus(expectation_value: jnp.ndarray) -> jnp.ndarray:
    """Convert quantum observable expectation value to probability of measuring -1.

    For a binary quantum observable $\\hat{O}$ with eigenvalues $b = \\{-1, 1\\}$, this function
    calculates the probability of measuring the eigenvalue -1 given its expectation value.

    Derivation:
    $$
        \\langle \\hat{O} \\rangle = -1 \\cdot \\Pr(b=-1) + 1 \\cdot \\Pr(b = 1)
    $$
        With the constraint $\\Pr(b = -1) + \\Pr(b = 1) = 1$:

    $$
        \\langle \\hat{O} \\rangle = -1 \\cdot \\Pr(b=-1) + 1 \\cdot (1 - \\Pr(b=-1)) \\
        \\langle \\hat{O} \\rangle = -\\Pr(b=-1) + 1 - \\Pr(b=-1) \\
        \\langle \\hat{O} \\rangle = 1 - 2\\Pr(b=-1) \\
        \\Pr(b=-1) = \\frac{1 - \\langle \\hat{O} \\rangle}{2}
    $$

    Args:
        expectation_value (jnp.ndarray): Expectation value of the quantum observable,
            must be in range [-1, 1].

    Returns:
        jnp.ndarray: Probability of measuring the -1 eigenvalue.
    """
    return (1 - expectation_value) / 2


def expectation_value_to_eigenvalue(
    expectation_value: jnp.ndarray, SHOTS: int
) -> jnp.ndarray:
    """Convert expectation value to eigenvalue

    Args:
        expectation_value (jnp.ndarray): Expectation value of quantum observable
        SHOTS (int): The number of shots used to produce expectation value

    Returns:
        jnp.ndarray: Array of eigenvalues
    """
    return jnp.where(
        jnp.broadcast_to(jnp.arange(SHOTS), expectation_value.shape + (SHOTS,))
        < jnp.around(
            expectation_value_to_prob_plus(
                jnp.reshape(expectation_value, expectation_value.shape + (1,))
            )
            * SHOTS
        ).astype(jnp.int32),
        1,
        -1,
    ).astype(jnp.int32)


def eigenvalue_to_binary(eigenvalue: jnp.ndarray) -> jnp.ndarray:
    """Convert -1 to 1, and 0 to 1
    This implementation should be differentiable

    Args:
        eigenvalue (jnp.ndarray): Eigenvalue to convert to bit value

    Returns:
        jnp.ndarray: Binary array
    """

    return (-1 * eigenvalue + 1) / 2


def binary_to_eigenvalue(binary: jnp.ndarray) -> jnp.ndarray:
    """Convert 1 to -1, and 0 to 1
    This implementation should be differentiable

    Args:
        binary (jnp.ndarray): Bit value to convert to eigenvalue

    Returns:
        jnp.ndarray: Eigenvalue array
    """

    return -1 * (binary * 2 - 1)


def get_dataset_metrics(
    loaded_data: LoadedData,
    NUM_EPOCH: int = 1000,
) -> DatasetMetrics:
    # * Data variance
    var = variance_of_observable(
        loaded_data.observed_values,
        shots=loaded_data.experiment_data.experiment_config.shots,
    ).mean()

    # * The MSE between ideal and experimental expectation values
    # Calculate the ideal expectation values of the original pulse
    ideal_expectation_values = calculate_expectation_values(loaded_data.unitaries)
    mse_ideal2exp = jax.vmap(mse, in_axes=(0, 0))(
        loaded_data.observed_values, ideal_expectation_values
    ).mean()

    # * The training iteration.
    total_iterations = 9 * NUM_EPOCH
    step_for_optimizer = 8 * NUM_EPOCH
    warmup_steps = int(0.1 * step_for_optimizer)
    cool_down_steps = total_iterations - step_for_optimizer

    return DatasetMetrics(
        var=var.item(),
        mse_ideal2exp=mse_ideal2exp.item(),
        total_iterations=total_iterations,
        step_for_optimizer=step_for_optimizer,
        warmup_steps=warmup_steps,
        cool_down_steps=cool_down_steps,
    )


def recursive_vmap(func, in_axes):
    """Perform recursive vmap on the given axis

    Note:
        ```python
        def func(x):
            assert x.ndim == 1
            return x ** 2
        x = jnp.arange(10)
        x_test = jnp.broadcast_to(x, (2, 3, 4,) + x.shape)
        x_test.shape, recursive_vmap(func, (0,) * (x_test.ndim - 1))(x_test).shape
        ((2, 3, 4, 10), (2, 3, 4, 10))
        ```

    Examples:
        >>> def func(x):
        ...     assert x.ndim == 1
        ...     return x ** 2
        >>> x = jnp.arange(10)
        >>> x_test = jnp.broadcast_to(x, (2, 3, 4,) + x.shape)
        >>> x_test.shape, recursive_vmap(func, (0,) * (x_test.ndim - 1))(x_test).shape
        ((2, 3, 4, 10), (2, 3, 4, 10))

    Args:
        func (typing.Any): The function for vmap
        in_axes (typing.Any): The axes for vmap

    Returns:
        typing.Any: _description_
    """
    if not in_axes:
        # Base case: no more axes to vectorize over
        return func

    # Apply vmap over the first axis specified in in_axes
    vmap_func = jax.vmap(func, in_axes=in_axes[0])

    # Recursively apply vmap over the remaining axes
    return recursive_vmap(vmap_func, in_axes[1:])


class SyntheticDataModel(typing.NamedTuple):
    control_sequence: ControlSequence
    qubit_information: QubitInformation
    dt: float
    ideal_hamiltonian: typing.Callable[..., jnp.ndarray]
    total_hamiltonian: typing.Callable[..., jnp.ndarray]
    solver: typing.Callable[..., jnp.ndarray]
    quantum_device: typing.Callable[..., jnp.ndarray] | None
    whitebox: typing.Callable[..., jnp.ndarray] | None


def calculate_shots_expectation_value(
    key: jnp.ndarray,
    initial_state: jnp.ndarray,
    unitary: jnp.ndarray,
    operator: jnp.ndarray,
    shots: int,
) -> jnp.ndarray:
    """Calculate finite-shots estimate of expectation value

    Args:
        key (jnp.ndarray): Random key
        initial_state (jnp.ndarray): Inital state
        unitary (jnp.ndarray): Unitary operator
        plus_projector (jnp.ndarray): The eigenvector corresponded to +1 eigenvalue of Pauli observable.
        shots (int): Number of shot to be used in estimation of expectation value

    Returns:
        jnp.ndarray: Finite-shot estimate expectation value
    """
    expval = jnp.trace(unitary @ initial_state @ unitary.conj().T @ operator).real
    prob = expectation_value_to_prob_plus(expval)

    return jax.random.choice(
        key, jnp.array([1, -1]), shape=(shots,), p=jnp.array([prob, 1 - prob])
    ).mean()


def shot_quantum_device(
    key: jnp.ndarray,
    control_parameters: jnp.ndarray,
    solver: typing.Callable[[jnp.ndarray], jnp.ndarray],
    SHOTS: int,
    expectation_value_receipt: typing.Sequence[
        ExpectationValue
    ] = default_expectation_values_order,
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

    expectation_values = jnp.zeros((control_parameters.shape[0], 18))
    unitaries = jax.vmap(solver)(control_parameters)[:, -1, :, :]

    for idx, exp in enumerate(expectation_value_receipt):
        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, num=unitaries.shape[0])

        expectation_value = jax.vmap(
            calculate_shots_expectation_value,
            in_axes=(0, None, 0, None, None),
        )(
            sample_keys,
            exp.initial_density_matrix,
            unitaries,
            exp.observable_matrix,
            SHOTS,
        )

        expectation_values = expectation_values.at[..., idx].set(expectation_value)

    return expectation_values


def get_spam(params: VariableDict):
    pair_map = {"+": "-", "-": "+", "0": "1", "1": "0", "r": "l", "l": "r"}
    observables = {"X": X, "Y": Y, "Z": Z}
    for pauli, matrix in observables.items():
        p_10 = params["AM"][pauli]["prob_10"]
        p_01 = params["AM"][pauli]["prob_01"]

        observables[pauli] = (
            matrix
            + (-2 * p_10 * plus_projectors[pauli])
            + (2 * p_01 * minus_projectors[pauli])
        )

    expvals = []
    order_expvals = get_default_expectation_values_order()
    for _expval in order_expvals:
        expval = ExpectationValue(
            initial_state=_expval.initial_state, observable=_expval.observable
        )
        # SP State Preparation error
        SP_correct_prob = params["SP"][_expval.initial_state]
        SP_incorrect_prob = 1 - SP_correct_prob
        expval.initial_density_matrix = SP_correct_prob * State.from_label(
            _expval.initial_state, dm=True
        ) + SP_incorrect_prob * State.from_label(
            pair_map[_expval.initial_state], dm=True
        )
        # AM, And Measurement error
        expval.observable_matrix = observables[_expval.observable]

        expvals.append(expval)

    return expvals, observables


def parse_expectation_values(
    expectation_values: jnp.ndarray,
    expectation_value_receipt: list[
        ExpectationValue
    ] = default_expectation_values_order,
) -> list[ExpectationValue]:
    return [
        ExpectationValue(
            initial_state=exp.initial_state,
            observable=exp.observable,
            expectation_value=expval.item(),
        )
        for exp, expval in zip(expectation_value_receipt, expectation_values)
    ]


def enable_jax_x64():
    jax.config.update("jax_enable_x64", True)


def disable_jax_x64():
    jax.config.update("jax_enable_x64", False)
