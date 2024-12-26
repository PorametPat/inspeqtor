import jax
import jax.numpy as jnp
from numpyro import handlers
import optax
from alive_progress import alive_it
import typing
import jax
import jax.numpy as jnp
import inspeqtor.experimental as sq

def gate_optimizer(
    params,
    lower,
    upper,
    func: typing.Callable,
    optimizer: optax.GradientTransformation,
    maxiter: int = 1000,
):

    opt_state = optimizer.init(params)
    history = []

    for _ in alive_it(range(maxiter), force_tty=True):
        grads, aux = jax.grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Apply projection
        params = optax.projections.projection_box(params, lower, upper)

        # Log the history
        aux['params'] = params
        history.append(aux)

    return params, history

def detune_x_hamiltonian(
    hamiltonian: typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    detune: float,
) -> typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
    def detuned_hamiltonian(
        params: sq.typing.HamiltonianArgs,
        t: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return hamiltonian(params, t, *args, **kwargs) + detune * sq.constant.X

    return detuned_hamiltonian

def safe_shape(a):
    try:
        return a.shape
    except AttributeError:
        return type(a)


def report_shape(a):
    return jax.tree.map(safe_shape, a)


def lexpand(a: jnp.ndarray, *dimensions: int) -> jnp.ndarray:
    """Expand tensor, adding new dimensions on left."""
    return jnp.broadcast_to(a, dimensions + a.shape)


# Custom random split function
def random_split_index(rng_key, num_samples, test_size):
    idx = jax.random.permutation(rng_key, jnp.arange(num_samples))
    return idx[test_size:], idx[:test_size]


def _safe_mean_terms(terms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    nonnan = jnp.isfinite(terms)
    terms = jnp.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    loss = terms.sum(axis=0) / nonnan.sum(axis=0)
    loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    return loss.sum(), loss


def _safe_mean_terms_v2(terms: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    nonnan = jnp.isfinite(terms)
    terms = jnp.nan_to_num(terms)
    loss = terms.sum(axis=0) / nonnan.sum(axis=0)
    loss = jnp.nan_to_num(loss)

    return loss.sum(), loss


def test_safe_mean_terms():
    # Test cases
    test_cases = [
        jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        jnp.array([1.0, jnp.nan, 3.0], dtype=jnp.float32),
        jnp.array([jnp.inf, 2.0, -jnp.inf], dtype=jnp.float32),
        jnp.array([jnp.nan, jnp.nan, jnp.nan], dtype=jnp.float32),  # NOTE: Fail.
        jnp.array([1.0, 2.0, 3.0, -jnp.inf, jnp.nan], dtype=jnp.float64),
    ]

    for i, terms in enumerate(test_cases):
        agg_loss, loss = _safe_mean_terms(terms)
        mask = jnp.isnan(terms) | (terms == float("-inf")) | (terms == float("inf"))
        print(f"Test case {i+1}:")
        print(f"mask: {mask}")
        print(f"Input terms: {terms}")
        print(f"Aggregate loss: {agg_loss}")
        print(f"Loss: {loss}\n")


def marginal_loss(
    model,
    marginal_guide,
    design,
    *args,
    observation_labels,
    target_labels,
    num_particles,
    evaluation=False,
):
    # Marginal loss
    def loss_fn(param, key):
        expanded_design = lexpand(design, num_particles)

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
        terms = jnp.array(
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
        return agg_loss, {"terms": terms, "eig": loss}

    return loss_fn


def init_params_from_guide(
    marginal_guide,
    *args,
    key: jnp.ndarray,
    design: jnp.ndarray,
    num_particles: int,
):
    key, subkey = jax.random.split(key)
    expanded_design = lexpand(design, num_particles)
    marginal_guide_trace = handlers.trace(
        handlers.seed(marginal_guide, subkey)
    ).get_trace(expanded_design, *args, observation_labels=[], target_labels=[])

    # Get only nodes that are parameters
    params = {
        name: node["value"]
        for name, node in marginal_guide_trace.items()
        if node["type"] == "param"
    }

    return params


def opt_eig_ape_loss(
    loss_fn,
    params,
    num_steps: int,
    optim: optax.GradientTransformation,
    key: jnp.ndarray,
):
    # Initialize the optimizer
    opt_state = optim.init(params)
    # jit the loss function
    loss_fn = jax.jit(loss_fn)

    history = []
    for step in alive_it(range(num_steps), force_tty=True):
        key, subkey = jax.random.split(key)
        # Compute the loss and its gradient
        (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey)
        # Update the optimizer and params
        updates, opt_state = optim.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        history.append((step, loss, aux))

    return params, history


def marginal_eig(
    key: jnp.ndarray,
    model,
    marginal_guide,
    design: jnp.ndarray,
    *args,
    optimizer: optax.GradientTransformation,
    num_optimization_steps: int,
    observation_labels: list[str],
    target_labels: list[str],
    num_particles: int,
    final_num_particles: int | None = None,
):
    # NOTE: In final evalution, if final_num_particles != num_particles,
    # the code will error because we train params with num_particles
    # the shape will mismatch
    final_num_particles = final_num_particles or num_particles

    # Initialize the parameters by using trace from the marginal_guide
    key, subkey = jax.random.split(key)
    params = init_params_from_guide(
        marginal_guide, *args, key=subkey, design=design, num_particles=num_particles
    )

    # Optimize the loss function first to get the optimal parameters
    # for marginal guide
    params, history = opt_eig_ape_loss(
        loss_fn=marginal_loss(
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
    )

    key, subkey = jax.random.split(key)
    # Evaluate the loss
    _, aux = marginal_loss(
        model,
        marginal_guide,
        design,
        *args,
        observation_labels=observation_labels,
        target_labels=target_labels,
        num_particles=final_num_particles,
        evaluation=True,
    )(params, subkey)

    return aux["eig"], {
        "params": params,
        "history": history,
    }


def get_default_optimizer(n_iterations):
    return optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=int(0.1 * n_iterations),
            decay_steps=n_iterations,
            end_value=1e-6,
        )
    )
