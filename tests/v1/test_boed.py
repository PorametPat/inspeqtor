import jax.numpy as jnp
from inspeqtor.v1.boed import (
    _safe_mean_terms,
    make_marginal_guide_from_model_v2,
)
import jax
import numpyro
import numpyro.distributions as dist
from inspeqtor.v1.boed import (
    marginal_loss,
    init_params_from_guide,
    # make_marginal_guide_from_model,
)


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
        print(f"Test case {i + 1}:")
        print(f"mask: {mask}")
        print(f"Input terms: {terms}")
        print(f"Aggregate loss: {agg_loss}")
        print(f"Loss: {loss}\n")


# 1. Define the Gaussian Model
def linear_gaussian_model(design, obs_noise=0.1):
    # Expand prior shapes so they are naturally vectorized when design is multi-dimensional
    theta = numpyro.sample(
        "theta", dist.Normal(jnp.zeros_like(design), jnp.ones_like(design))
    )
    numpyro.sample("y", dist.Normal(theta * design, obs_noise))


def make_marginal_guide(original_design_shape):
    """Factory that locks in the true design dimensions upfront."""

    def marginal_guide(
        design, obs_noise=0.1, observation_labels=None, target_labels=None
    ):
        # Parameters match the true design shape, completely immune to particle expansion
        loc = numpyro.param("loc", jnp.zeros(original_design_shape))
        scale = numpyro.param(
            "scale",
            jnp.ones(original_design_shape),
            constraint=dist.constraints.positive,
        )

        assert isinstance(loc, jnp.ndarray) and isinstance(scale, jnp.ndarray)

        # Dynamically broadcast parameters up to whatever design shape is passed in
        # (Works perfectly whether design is un-expanded or expanded with particles!)
        loc_b = jnp.broadcast_to(loc, design.shape)
        scale_b = jnp.broadcast_to(scale, design.shape)

        numpyro.sample("y", dist.Normal(loc_b, scale_b))

    return marginal_guide


def test_marginal_loss_analytic_baseline():
    design = jnp.array([1.0])
    obs_noise = 0.5
    true_eig = 0.5 * jnp.log(1.0 + (design**2 / obs_noise**2))

    # Instantiate the robust guide matching the design shape
    # marginal_guide = make_marginal_guide(design.shape)
    marginal_guide = make_marginal_guide_from_model_v2(
        linear_gaussian_model, design, key=jax.random.key(0)
    )

    loss_fn = marginal_loss(
        linear_gaussian_model,
        marginal_guide,
        design,
        obs_noise,
        observation_labels=["y"],
        target_labels=["theta"],
        num_particles=5000,
        evaluation=True,
    )

    rng_key = jax.random.key(0)
    params = init_params_from_guide(
        marginal_guide, obs_noise, key=rng_key, design=design, observation_labels=[]
    )

    params["q_y_loc"] = jnp.array([0.0])  # type: ignore
    params["q_y_scale"] = jnp.sqrt(design**2 + obs_noise**2)  # type: ignore

    agg_loss, aux = loss_fn(params, rng_key)
    estimated_eig = aux.eig

    assert jnp.allclose(estimated_eig, true_eig, atol=0.05)
