import jax
import jax.numpy as jnp
from numpyro import handlers, plate_stack  # type: ignore

import optax  # type: ignore
from alive_progress import alive_it  # type: ignore
import typing
import jaxtyping
import chex
from .decorator import warn_not_tested_function


def safe_shape(a: typing.Any) -> tuple[int, ...] | str:
    """Safely get the shape of the object

    Args:
        a (typing.Any): Expect the object to be jnp.ndarray

    Returns:
        tuple[int, ...] | str: Either return the shape of `a`
        or string representation of the type
    """
    try:
        assert isinstance(a, jnp.ndarray)
        return a.shape
    except AttributeError:
        return str(type(a))


def report_shape(a: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Report the shape of pytree

    Args:
        a (jaxtyping.PyTree): The pytree to be report.

    Returns:
        jaxtyping.PyTree: The shape of pytree.
    """
    return jax.tree.map(safe_shape, a)


def lexpand(a: jnp.ndarray, *dimensions: int) -> jnp.ndarray:
    """Expand tensor, adding new dimensions on left.

    Args:
        a (jnp.ndarray): expand the dimension on the left with given dimension arguments.

    Returns:
        jnp.ndarray: New array with shape (*dimension + a.shape)
    """
    return jnp.broadcast_to(a, dimensions + a.shape)


def random_split_index(
    rng_key: jnp.ndarray, num_samples: int, test_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create the randomly spilt of indice to two set, with one of test_size and another as the rest.

    Args:
        rng_key (jnp.ndarray): The random key
        num_samples (int): The size of total sample size
        test_size (int): The size of test set

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Array of train indice and array of test indice.
    """
    idx = jax.random.permutation(rng_key, jnp.arange(num_samples))
    return idx[test_size:], idx[:test_size]


def _safe_mean_terms(terms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Safely calculate the mean of the first axis.

    Args:
        terms (jnp.ndarray): Multi axis jnp.ndarray to calculate the mean value

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: sum of all value in the first axis-averaged array, first axis-averaged array
    """
    nonnan = jnp.isfinite(terms)
    terms = jnp.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    loss = terms.sum(axis=0) / nonnan.sum(axis=0)
    loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss.sum(), loss


def _safe_mean_terms_v2(terms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Safely calculate the mean of the first axis.
       This is the simplify version

    Args:
        terms (jnp.ndarray): Multi axis jnp.ndarray to calculate the mean value

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: sum of all value in the first axis-averaged array, first axis-averaged array
    """
    nonnan = jnp.isfinite(terms)
    terms = jnp.nan_to_num(terms)
    loss = terms.sum(axis=0) / nonnan.sum(axis=0)
    loss = jnp.nan_to_num(loss)

    return loss.sum(), loss


class AuxEntry(typing.NamedTuple):
    """The auxillary entry returned by loss function

    Args:
        terms (jnp.ndarray): _description_
        eig (jnp.ndarray): Expected information gain
    """

    terms: jnp.ndarray | None
    eig: jnp.ndarray


def vectorized(fn, *shape, name="vectorization_plate"):
    def wrapper_fn(*args, **kwargs):
        with plate_stack(name, [*shape]):
            return fn(*args, **kwargs)

    return wrapper_fn


def marginal_loss(
    model: typing.Callable,
    marginal_guide: typing.Callable,
    design: jnp.ndarray,
    *args,
    observation_labels: list[str],
    target_labels: list[str],
    num_particles: int,
    evaluation: bool = False,
) -> typing.Callable[[chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry]]:
    """The marginal loss implemented following
    https://docs.pyro.ai/en/dev/contrib.oed.html#pyro.contrib.oed.eig.marginal_eig

    Args:
        model (typing.Callable): The probabilistic model
        marginal_guide (typing.Callable): The custom guide
        design (jnp.ndarray): Possible designs of the experiment
        observation_labels (list[str]): The list of string of observations
        target_labels (list[str]): The target latent parameters to be optimized for
        num_particles (int): The number of independent trials
        evaluation (bool, optional): True for actual evalution of the EIG. Defaults to False.

    Returns:
        typing.Callable[ [chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry] ]: Loss function that return tuple of (1) Total loss, (2.1) Each terms without the average, (2.2) The EIG
    """

    # Marginal loss
    def loss_fn(param, key: jnp.ndarray) -> tuple[jnp.ndarray, AuxEntry]:
        expanded_design = lexpand(design, num_particles)
        # vectorized(model, num_particles)
        # Sample from p(y | d)
        key, subkey = jax.random.split(key)
        trace = handlers.trace(handlers.seed(model, subkey)).get_trace(
            expanded_design,
            *args,
        )
        y_dict = {
            observation_label: trace[observation_label]["value"]
            for observation_label in observation_labels
        }

        # Run through q(y | d)
        key, subkey = jax.random.split(key)
        conditioned_marginal_guide = handlers.condition(marginal_guide, data=y_dict)
        cond_trace = handlers.trace(
            handlers.substitute(
                handlers.seed(conditioned_marginal_guide, subkey), data=param
            )
        ).get_trace(
            expanded_design,
            *args,
            observation_labels=observation_labels,
            target_labels=target_labels,
        )
        # Compute the log prob of observing the data
        terms = -1 * jnp.array(
            [
                cond_trace[observation_label]["fn"].log_prob(
                    cond_trace[observation_label]["value"]
                )
                for observation_label in observation_labels
            ]
        ).sum(axis=0)

        if evaluation:
            terms += jnp.array(
                [
                    trace[observation_label]["fn"].log_prob(
                        trace[observation_label]["value"]
                    )
                    for observation_label in observation_labels
                ]
            ).sum(axis=0)

        agg_loss, loss = _safe_mean_terms_v2(terms)
        # return agg_loss, AuxEntry(terms=None, eig=None)
        # return agg_loss, {"terms": terms, "eig": loss}
        return agg_loss, AuxEntry(terms=None, eig=loss)

    return loss_fn


@warn_not_tested_function
def vnmc_eig_loss(
    model: typing.Callable,
    marginal_guide: typing.Callable,
    design: jnp.ndarray,
    *args,
    observation_labels: list[str],
    target_labels: list[str],
    num_particles: tuple[int, int],
    evaluation: bool = False,
) -> typing.Callable[[chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry]]:
    """The VNMC loss implemented following
    https://docs.pyro.ai/en/dev/_modules/pyro/contrib/oed/eig.html#vnmc_eig

    Args:
        model (typing.Callable): The probabilistic model
        marginal_guide (typing.Callable): The custom guide
        design (jnp.ndarray): Possible designs of the experiment
        observation_labels (list[str]): The list of string of observations
        target_labels (list[str]): The target latent parameters to be optimized for
        num_particles (int): The number of independent trials
        evaluation (bool, optional): True for actual evalution of the EIG. Defaults to False.

    Returns:
        typing.Callable[ [chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry] ]: Loss function that return tuple of (1) Total loss, (2.1) Each terms without the average, (2.2) The EIG
    """

    # Marginal loss
    def loss_fn(param, key: jnp.ndarray) -> tuple[jnp.ndarray, AuxEntry]:
        N, M = num_particles

        expanded_design = lexpand(design, N)

        # Sample from p(y, theta | d)
        key, subkey = jax.random.split(key)
        trace = handlers.trace(handlers.seed(model, subkey)).get_trace(
            expanded_design,
            *args,
        )
        y_dict = {
            observation_label: trace[observation_label]["value"]
            for observation_label in observation_labels
        }

        # Sample M times from q(theta | y, d) for each y
        key, subkey = jax.random.split(key)
        reexpanded_design = lexpand(expanded_design, M)
        conditioned_marginal_guide = handlers.condition(marginal_guide, data=y_dict)
        cond_trace = handlers.trace(
            handlers.substitute(
                handlers.seed(conditioned_marginal_guide, subkey), data=param
            )
        ).get_trace(
            reexpanded_design,
            *args,
            observation_labels=observation_labels,
            target_labels=target_labels,
        )

        theta_y_dict = {
            target_label: cond_trace[target_label]["value"]
            for target_label in target_labels
        }
        theta_y_dict.update(y_dict)

        # Re-run that through the model to compute the joint
        key, subkey = jax.random.split(key)
        conditioned_model = handlers.condition(model, data=theta_y_dict)
        conditioned_model_trace = handlers.trace(
            handlers.seed(conditioned_model, subkey)
        ).get_trace(
            reexpanded_design,
            *args,
        )

        # Compute the log prob of observing the data
        terms = -1 * jnp.array(
            [
                cond_trace[target_label]["fn"].log_prob(
                    cond_trace[target_label]["value"]
                )
                for target_label in target_labels
            ]
        ).sum(axis=0)

        terms += jnp.array(
            [
                conditioned_model_trace[target_label]["fn"].log_prob(
                    conditioned_model_trace[target_label]["value"]
                )
                for target_label in target_labels
            ]
        ).sum(axis=0)

        terms += jnp.array(
            [
                conditioned_model_trace[observation_label]["fn"].log_prob(
                    conditioned_model_trace[observation_label]["value"]
                )
                for observation_label in observation_labels
            ]
        ).sum(axis=0)

        terms = -jax.scipy.special.logsumexp(terms, axis=0) + jnp.log(M)

        if evaluation:
            terms += jnp.array(
                [
                    trace[observation_label]["fn"].log_prob(
                        trace[observation_label]["value"]
                    )
                    for observation_label in observation_labels
                ]
            ).sum(axis=0)

        agg_loss, loss = _safe_mean_terms_v2(terms)
        # return agg_loss, {"terms": terms, "eig": loss}
        return agg_loss, AuxEntry(terms=terms, eig=loss)

    return loss_fn


def init_params_from_guide(
    marginal_guide: typing.Callable,
    *args,
    key: jnp.ndarray,
    design: jnp.ndarray,
    # num_particles: int,
) -> chex.ArrayTree:
    """Initlalize parameters of marginal guide.

    Args:
        marginal_guide (typing.Callable): Marginal guide to be used with marginal eig
        key (jnp.ndarray): Random Key
        design (jnp.ndarray): Example of the designs of the experiment
        num_particles (int): Number of independent trials

    Returns:
        chex.ArrayTree: Random parameters for marginal guide to be optimized.
    """
    key, subkey = jax.random.split(key)
    # expanded_design = lexpand(design, num_particles)
    marginal_guide_trace = handlers.trace(
        handlers.seed(marginal_guide, subkey)
    ).get_trace(design, *args, observation_labels=[], target_labels=[])

    # Get only nodes that are parameters
    params = {
        name: node["value"]
        for name, node in marginal_guide_trace.items()
        if node["type"] == "param"
    }

    return params


class HistoryEntry(typing.NamedTuple):
    step: int
    loss: jnp.ndarray
    aux: AuxEntry


def opt_eig_ape_loss(
    loss_fn: typing.Callable[
        [chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry]
    ],
    params: chex.ArrayTree,
    num_steps: int,
    optim: optax.GradientTransformation,
    key: jnp.ndarray,
    progress: bool = True,
    callbacks: list = [],
) -> tuple[chex.ArrayTree, list[typing.Any]]:
    """Optimize the EIG loss function.

    Args:
        loss_fn (typing.Callable[[chex.ArrayTree, jnp.ndarray], tuple[jnp.ndarray, AuxEntry]]): Loss function
        params (chex.ArrayTree): Initial parameter
        num_steps (int): Number of optimization step
        optim (optax.GradientTransformation): Optax Optimizer
        key (jnp.ndarray): Random key

    Returns:
        tuple[chex.ArrayTree, list[typing.Any]]: Optimized parameters, and optimization history.
    """
    # Initialize the optimizer
    opt_state = optim.init(params)
    # jit the loss function
    loss_fn = jax.jit(loss_fn)

    iterator = (
        alive_it(range(num_steps), force_tty=True) if progress else range(num_steps)
    )

    history = []
    for step in iterator:
        key, subkey = jax.random.split(key)
        # Compute the loss and its gradient
        (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey)
        # Update the optimizer and params
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        history.append((step, loss, aux))

        for callback in callbacks:
            callback(history[-1])

    return params, history


def estimate_eig(
    key: jnp.ndarray,
    model: typing.Callable,
    marginal_guide: typing.Callable,
    design: jnp.ndarray,
    *args,
    optimizer: optax.GradientTransformation,
    num_optimization_steps: int,
    observation_labels: list[str],
    target_labels: list[str],
    num_particles: tuple[int, int] | int,
    final_num_particles: tuple[int, int] | int | None = None,
    loss_fn: typing.Callable = marginal_loss,
    progress: bool = True,
    callbacks: list = [],
) -> tuple[jnp.ndarray, dict[str, typing.Any]]:
    """Optimize for marginal EIG

    Args:
        key (jnp.ndarray): Random key
        model (typing.Callable): Probabilistic model of the experiment
        marginal_guide (typing.Callable): The marginal guide of the experiment
        design (jnp.ndarray): Possible designs of the experiment
        optimizer (optax.GradientTransformation): Optax optimizer
        num_optimization_steps (int): Number of the optimization step
        observation_labels (list[str]): The list of string of observations
        target_labels (list[str]): The target latent parameters to be optimized for
        num_particles (int): The number of independent trials
        final_num_particles (int | None, optional): Final independent trials to calculate marginal EIG. Defaults to None.

    Returns:
        tuple[jnp.ndarray, dict[str, typing.Any]]: EIG, and tuple of optimized parameters and optimization history.
    """
    # NOTE: In final evalution, if final_num_particles != num_particles,
    # the code will error because we train params with num_particles
    # the shape will mismatch
    # final_num_particles = final_num_particles or num_particles

    # Initialize the parameters by using trace from the marginal_guide
    key, subkey = jax.random.split(key)
    params = init_params_from_guide(
        marginal_guide,
        *args,
        key=subkey,
        design=design,
        # , num_particles=num_particles
    )

    # Optimize the loss function first to get the optimal parameters
    # for marginal guide
    params, history = opt_eig_ape_loss(
        loss_fn=loss_fn(
            model,
            marginal_guide,
            design,
            *args,
            observation_labels=observation_labels,
            target_labels=target_labels,
            num_particles=num_particles,
            evaluation=False,
        ),
        params=params,
        num_steps=num_optimization_steps,
        optim=optimizer,
        key=subkey,
        progress=progress,
        callbacks=callbacks,
    )

    key, subkey = jax.random.split(key)
    # Evaluate the loss
    _, aux = loss_fn(
        model,
        marginal_guide,
        design,
        *args,
        observation_labels=observation_labels,
        target_labels=target_labels,
        num_particles=final_num_particles,
        evaluation=True,
    )(params, subkey)

    return aux.eig, {
        "params": params,
        "history": history,
    }


def vectorized_for_eig(model):
    """Vectorization function for the EIG function

    Args:
        model (_type_): Probabilistic model.
    """

    def wrapper(
        design: jnp.ndarray,
        *args,
        # unitaries: jnp.ndarray,
        # observables: jnp.ndarray | None = None,
    ):
        # This wrapper has the same call signature as the probabilistic graybox model
        # Expect the design to has shape == (extra, design, feature)
        with plate_stack(prefix="vectorized_plate", sizes=[*design.shape[:2]]):
            return model(design, *args)

    return wrapper
