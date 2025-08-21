import typing
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import flax.traverse_util as traverse_util
import optax  # type: ignore
from alive_progress import alive_it  # type: ignore

import chex


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


def stochastic_minimize(
    key: jnp.ndarray,
    params: chex.ArrayTree,
    func: typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, typing.Any]],
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
        key, _ = jax.random.split(key)
        grads, aux = jax.grad(func, has_aux=True)(params, key)
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
