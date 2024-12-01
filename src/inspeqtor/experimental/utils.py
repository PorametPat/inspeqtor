import jax
import jax.numpy as jnp
import typing
import optax
import jaxtyping
from dataclasses import dataclass
from .pulse import PulseSequence
from .data import ExperimentData
from .constant import Z
from .decorator import not_yet_tested
from .sq_typing import HamiltonianArgs
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


def center_location(num_of_pulse_in_dd: int, total_time_dt: int | float):
    center_locations = (
        jnp.array(
            [(k - 0.5) / num_of_pulse_in_dd for k in range(1, num_of_pulse_in_dd + 1)]
        )
        * total_time_dt
    )  # ideal CPMG pulse locations for x-axis
    return center_locations


def drag_envelope_v2(
    amp: float | jnp.ndarray,
    sigma: float | jnp.ndarray,
    beta: float | jnp.ndarray,
    center: float | jnp.ndarray,
    final_amp: float | jnp.ndarray = 1.0,
):
    # https://docs.quantum.ibm.com/api/qiskit/qiskit.pulse.library.Drag_class.rst#drag

    def g(t):
        return jnp.exp(-((t - center) ** 2) / (2 * sigma**2))

    def g_prime(t):
        return amp * (g(t) - g(-1)) / (1 - g(-1))

    def envelop(t):
        return final_amp * g_prime(t) * (1 + 1j * beta * (t - center) / sigma**2)

    return envelop


def calculate_exp(
    unitary: jnp.ndarray, operator: jnp.ndarray, density_matrix: jnp.ndarray
) -> jnp.ndarray:
    rho = jnp.matmul(
        unitary, jnp.matmul(density_matrix, unitary.conj().swapaxes(-2, -1))
    )
    temp = jnp.matmul(rho, operator)
    return jnp.real(jnp.sum(jnp.diagonal(temp, axis1=-2, axis2=-1), axis=-1))


@not_yet_tested
def detune_hamiltonian(
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    detune: float,
) -> typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
    def detuned_hamiltonian(params: HamiltonianArgs, t: jnp.ndarray) -> jnp.ndarray:
        return hamiltonian(params, t) + detune * Z

    return detuned_hamiltonian


def prepare_data(
    exp_data: ExperimentData,
    pulse_sequence: PulseSequence,
    whitebox: typing.Callable,
) -> LoadedData:
    logging.info(f"Loaded data from {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}")

    pulse_parameters = jnp.array(exp_data.parameters)
    # * Attempt to reshape the pulse_parameters to (size, features)
    if len(pulse_parameters.shape) == 3:
        pulse_parameters = pulse_parameters.reshape(
            pulse_parameters.shape[0],
            pulse_parameters.shape[1] * pulse_parameters.shape[2],
        )

    expectations = jnp.array(exp_data.get_expectation_values())
    unitaries = jax.vmap(whitebox)(pulse_parameters)

    logging.info(
        f"Finished preparing the data for the experiment {exp_data.experiment_config.EXPERIMENT_IDENTIFIER}"
    )

    return LoadedData(
        experiment_data=exp_data,
        pulse_parameters=pulse_parameters,
        unitaries=unitaries[:, -1, :, :],
        expectation_values=expectations,
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


def test_dataloader(DATA_SIZE: int = 100, BATCH_SIZE: int = 15, NUM_EPOCHS: int = 10):
    train_key = jax.random.key(0)

    x_mock = jnp.linspace(0, 10, DATA_SIZE).reshape(-1, 1)
    y_mock = jnp.sin(x_mock)

    # Expected number of batches per epoch
    num_batches = x_mock.shape[0] // BATCH_SIZE
    if x_mock.shape[0] % BATCH_SIZE != 0:
        num_batches += 1

    expected_final_batch_idx = num_batches - 1
    expected_step = num_batches * NUM_EPOCHS - 1

    step = 0
    batch_idx = 0
    is_last_batch = True
    epoch_idx = 0

    for (step, batch_idx, is_last_batch, epoch_idx), (x_batch, y_batch) in dataloader(
        (x_mock, y_mock), batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, key=train_key
    ):
        print(
            f"step: {step}, batch_idx: {batch_idx}, is_last_batch: {is_last_batch}, epoch_idx: {epoch_idx}, x_batch: {x_batch.shape}, y_batch: {y_batch.shape}"
        )

    assert step == expected_step
    assert batch_idx == expected_final_batch_idx
    assert is_last_batch
    assert epoch_idx == NUM_EPOCHS - 1


def create_step(
    optimizer: optax.GradientTransformation,
    loss_fn: typing.Callable[..., jnp.ndarray],
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
