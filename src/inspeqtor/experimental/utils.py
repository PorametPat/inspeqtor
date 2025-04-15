import jax
import jax.numpy as jnp
import typing
import optax  # type: ignore
import jaxtyping
from dataclasses import dataclass
from .pulse import PulseSequence
from .data import ExperimentData, QubitInformation
from .model import mse
from .constant import Z, default_expectation_values_order
from .decorator import warn_not_tested_function
from .typing import HamiltonianArgs
from .physics import calculate_exp
import logging


@dataclass
class LoadedData:
    experiment_data: ExperimentData
    pulse_parameters: jnp.ndarray
    unitaries: jnp.ndarray
    expectation_values: jnp.ndarray
    pulse_sequence: PulseSequence
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
    pulse_sequence: PulseSequence,
    whitebox: typing.Callable,
) -> LoadedData:
    """Prepare the data for easy accessing from experiment data, control sequence, and Whitebox.

    Args:
        exp_data (ExperimentData): `ExperimentData` instance
        pulse_sequence (PulseSequence): Control sequence of the experiment
        whitebox (typing.Callable): Ideal unitary solver.

    Returns:
        LoadedData: `LoadedData` instance
    """
    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    pulse_parameters = jnp.array(exp_data.parameters)
    # * Attempt to reshape the pulse_parameters to (size, features)
    if len(pulse_parameters.shape) == 3:
        pulse_parameters = pulse_parameters.reshape(
            pulse_parameters.shape[0],
            pulse_parameters.shape[1] * pulse_parameters.shape[2],
        )

    expectation_values = jnp.array(exp_data.get_expectation_values())
    unitaries = jax.vmap(whitebox)(pulse_parameters)

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    return LoadedData(
        experiment_data=exp_data,
        pulse_parameters=pulse_parameters,
        unitaries=unitaries[:, -1, :, :],
        expectation_values=expectation_values,
        pulse_sequence=pulse_sequence,
        whitebox=whitebox,
    )


def random_split(key: jnp.ndarray, test_size: int, *data_arrays: jnp.ndarray):
    """The random_split function splits the data into training and testing sets.

    Example
    ```py
    key = jax.random.key(0)
    x = jnp.arange(10)
    y = jnp.arange(10)
    x_train, y_train, x_test, y_test = random_split(key, 2, x, y)
    assert x_train.shape[0] == 8 and y_train.shape[0] == 8
    assert x_test.shape[0] == 2 and y_test.shape[0] == 2
    ```

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
        _type_: (step, batch_idx, is_last_batch, epoch_idx), (array_batch, ...)
    """
    # * General dataloader
    # Check that all arrays have the same size in the first dimension
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
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


def create_step(
    optimizer: optax.GradientTransformation,
    loss_fn: (
        typing.Callable[..., jnp.ndarray]
        | typing.Callable[..., typing.Tuple[jnp.ndarray, typing.Any]]
    ),
    has_aux: bool = False,
):
    """The create_step function creates a training step function and a test step function.

    loss_fn should have the following signature:
    ```py
    def loss_fn(params: jaxtyping.PyTree, *args) -> jnp.ndarray:
        ...
        return loss_value
    ```
    where `params` is the parameters to be optimized, and `args` are the inputs for the loss function.

    Args:
        optimizer (optax.GradientTransformation): `optax` optimizer.
        loss_fn (typing.Callable[[jaxtyping.PyTree, ...], jnp.ndarray]): Loss function, which takes in the model parameters, inputs, and targets, and returns the loss value.
        has_aux (bool, optional): Whether the loss function return aux data or not. Defaults to False.

    Returns:
        __type__: train_step, test_step
    """

    # * Generalized training step
    @jax.jit
    def train_step(
        params: jaxtyping.PyTree,
        optimizer_state: optax.OptState,
        *args,
    ):
        loss_value, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(params, *args)
        updates, opt_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    @jax.jit
    def test_step(
        params: jaxtyping.PyTree,
        *args,
    ):
        return loss_fn(params, *args)

    return train_step, test_step


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


def get_dataset_metrics(
    loaded_data: LoadedData,
    NUM_EPOCH: int = 1000,
) -> DatasetMetrics:
    # * Data variance
    var = variance_of_observable(
        loaded_data.expectation_values,
        shots=loaded_data.experiment_data.experiment_config.shots,
    ).mean()

    # * The MSE between ideal and experimental expectation values
    # Calculate the ideal expectation values of the original pulse
    ideal_expectation_values = calculate_expectation_values(loaded_data.unitaries)
    mse_ideal2exp = jax.vmap(mse, in_axes=(0, 0))(
        loaded_data.expectation_values, ideal_expectation_values
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
    """
    Recursively apply jax.vmap over multiple dimensions.
    ```python
    def func(x):
        assert x.ndim == 1
        return x ** 2

    x = jnp.arange(10)
    x_test = jnp.broadcast_to(x, (2, 3, 4,) + x.shape)
    x_test.shape, recursive_vmap(func, (0,) * (x_test.ndim - 1))(x_test).shape
    >>> ((2, 3, 4, 10), (2, 3, 4, 10))
    ```

    Parameters:
    - func: The function to be vectorized.
    - in_axes: A tuple of integers specifying which axes to map over.

    Returns:
    - A new function that applies `func` vectorized over the specified axes.
    """
    if not in_axes:
        # Base case: no more axes to vectorize over
        return func

    # Apply vmap over the first axis specified in in_axes
    vmap_func = jax.vmap(func, in_axes=in_axes[0])

    # Recursively apply vmap over the remaining axes
    return recursive_vmap(vmap_func, in_axes[1:])


class SyntheticDataModel(typing.NamedTuple):
    pulse_sequence: PulseSequence
    qubit_information: QubitInformation
    dt: float
    ideal_hamiltonian: typing.Callable[..., jnp.ndarray]
    total_hamiltonian: typing.Callable[..., jnp.ndarray]
    solver: typing.Callable[..., jnp.ndarray]
    quantum_device: typing.Callable[..., jnp.ndarray] | None
