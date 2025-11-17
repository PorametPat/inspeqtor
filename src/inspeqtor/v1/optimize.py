import typing
import jax
import jax.numpy as jnp
import optax  # type: ignore
import chex


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
    callbacks: list[typing.Callable] = [],
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

    for step_idx in range(maxiter):
        grads, aux = jax.grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if lower is not None and upper is not None:
            # Apply projection
            params = optax.projections.projection_box(params, lower, upper)

        # Log the history
        aux["params"] = params
        history.append(aux)

        for callback in callbacks:
            callback(step_idx, aux)

    return params, history


def stochastic_minimize(
    key: jnp.ndarray,
    params: chex.ArrayTree,
    func: typing.Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, typing.Any]],
    optimizer: optax.GradientTransformation,
    lower: chex.ArrayTree | None = None,
    upper: chex.ArrayTree | None = None,
    maxiter: int = 1000,
    callbacks: list[typing.Callable] = [],
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

    for step_idx in range(maxiter):
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

        for callback in callbacks:
            callback(step_idx, aux)

    return params, history
