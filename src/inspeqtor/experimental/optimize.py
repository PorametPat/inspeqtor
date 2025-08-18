import typing
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
import flax.traverse_util as traverse_util
import optax  # type: ignore
from alive_progress import alive_it  # type: ignore

import chex
import jaxtyping

from .utils import dataloader
from .model import LossMetric, load_model


@jax.tree_util.register_dataclass
@dataclass
class DataBundled:
    control_params: jnp.ndarray
    unitaries: jnp.ndarray
    observables: jnp.ndarray
    aux: jnp.ndarray | None = None


def get_default_optimizer(n_iterations: int) -> optax.GradientTransformation:
    """Generate present optimizer from number of training iteration.

    Args:
        n_iterations (int): Training iteration

    Returns:
        optax.GradientTransformation: Optax optimizer.
    """
    return optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=int(0.1 * n_iterations),
            decay_steps=n_iterations,
            end_value=1e-6,
        )
    )


def minimize(
    params: chex.ArrayTree,
    func: typing.Callable[[jnp.ndarray], tuple[jnp.ndarray, typing.Any]],
    optimizer: optax.GradientTransformation,
    lower: chex.ArrayTree | None = None,
    upper: chex.ArrayTree | None = None,
    maxiter: int = 1000,
) -> tuple[chex.ArrayTree, list[typing.Any]]:
    """Optimize the loss function with bounded parameters.

    Args:
        params (chex.ArrayTree): Intiial parameters to be optimized
        lower (chex.ArrayTree): Lower bound of the parameters
        upper (chex.ArrayTree): Upper bound of the parameters
        func (typing.Callable[[jnp.ndarray], tuple[jnp.ndarray, typing.Any]]): Loss function
        optimizer (optax.GradientTransformation): Instance of optax optimizer
        maxiter (int, optional): Number of optimization step. Defaults to 1000.

    Returns:
        tuple[chex.ArrayTree, list[typing.Any]]: Tuple of parameters and optimization history
    """
    opt_state = optimizer.init(params)
    history = []

    for _ in alive_it(range(maxiter), force_tty=True):
        grads, aux = jax.grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if lower is not None and upper is not None:
            # Apply projection
            params = optax.projections.projection_box(params, lower, upper)

        # Log the history
        aux["params"] = params
        history.append(aux)

    return params, history


@dataclass
class HistoryEntryV3:
    step: int
    loss: float | jnp.ndarray
    loop: str
    aux: dict[str, jnp.ndarray]


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
        **kwargs,
    ):
        loss_value, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(
            params, *args, **kwargs
        )
        updates, opt_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    @jax.jit
    def test_step(
        params: jaxtyping.PyTree,
        *args,
        **kwargs,
    ):
        return loss_fn(params, *args, **kwargs)

    return train_step, test_step


def train_model(
    # Random key
    key: jnp.ndarray,
    # Data
    train_data: DataBundled,
    val_data: DataBundled,
    test_data: DataBundled,
    # Model to be used for training
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    # Loss function to be used
    loss_fn: typing.Callable,
    # Callbacks to be used
    callbacks: list[typing.Callable] = [],
    # Number of epochs
    NUM_EPOCH: int = 1_000,
    # Optional state
    model_params: typing.Any = None,
    opt_state: typing.Any = None,
):
    """Train the BlackBox model

    >>> # The number of epochs break down
    ... NUM_EPOCH = 150
    ... # Total number of iterations as 90% of data is used for training
    ... # 10% of the data is used for testing
    ... total_iterations = 9 * NUM_EPOCH
    ... # The step for optimizer if set to 8 * NUM_EPOCH (should be less than total_iterations)
    ... step_for_optimizer = 8 * NUM_EPOCH
    ... optimizer = get_default_optimizer(step_for_optimizer)
    ... # The warmup steps for the optimizer
    ... warmup_steps = 0.1 * step_for_optimizer
    ... # The cool down steps for the optimizer
    ... cool_down_steps = total_iterations - step_for_optimizer
    ... total_iterations, step_for_optimizer, warmup_steps, cool_down_steps

    Args:
        key (jnp.ndarray): Random key
        model (nn.Module): The model to be used for training
        optimizer (optax.GradientTransformation): The optimizer to be used for training
        loss_fn (typing.Callable): The loss function to be used for training
        callbacks (list[typing.Callable], optional): list of callback functions. Defaults to [].
        NUM_EPOCH (int, optional): The number of epochs. Defaults to 1_000.

    Returns:
        tuple: The model parameters, optimizer state, and the histories
    """

    key, loader_key, init_key = jax.random.split(key, 3)

    train_p, train_u, train_ex = (
        train_data.control_params,
        train_data.unitaries,
        train_data.observables,
    )
    val_p, val_u, val_ex = (
        val_data.control_params,
        val_data.unitaries,
        val_data.observables,
    )
    test_p, test_u, test_ex = (
        test_data.control_params,
        test_data.unitaries,
        test_data.observables,
    )

    BATCH_SIZE = val_p.shape[0]

    if model_params is None:
        # Initialize the model parameters if it is None
        model_params = model.init(init_key, train_p[0])

    if opt_state is None:
        # Initalize the optimizer state if it is None
        opt_state = optimizer.init(model_params)

    # histories: list[dict[str, typing.Any]] = []
    histories: list[HistoryEntryV3] = []

    train_step, eval_step = create_step(
        optimizer=optimizer, loss_fn=loss_fn, has_aux=True
    )

    for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (train_p, train_u, train_ex),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
        model_params, opt_state, (loss, aux) = train_step(
            model_params, opt_state, batch_p, batch_u, batch_ex
        )

        histories.append(HistoryEntryV3(step=step, loss=loss, loop="train", aux=aux))

        if is_last_batch:
            # Validation
            (val_loss, aux) = eval_step(model_params, val_p, val_u, val_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=val_loss, loop="val", aux=aux)
            )

            # Testing
            (test_loss, aux) = eval_step(model_params, test_p, test_u, test_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=test_loss, loop="test", aux=aux)
            )

            for callback in callbacks:
                callback(model_params, opt_state, histories)

    return model_params, opt_state, histories


def transform_key(data):
    return {
        # Concanate the key by '/'
        "/".join(key): value
        for key, value in data.items()
    }


def clean_history_entries(
    histories: list[HistoryEntryV3],
):
    clean_histories = [
        {
            "step": history.step,
            "loss": history.loss,
            "loop": history.loop,
            **history.aux,
        }
        for history in histories
    ]
    # Move from device to host, i.e. from jax.Array to numpy.ndarray
    clean_histories = jax.tree.map(
        lambda x: x.item() if isinstance(x, jnp.ndarray) else x, clean_histories
    )
    # Flatten the nested dictionaries
    clean_histories = list(map(traverse_util.flatten_dict, clean_histories))
    # Transform the keys of the dictionary
    clean_histories = list(map(transform_key, clean_histories))
    return clean_histories
