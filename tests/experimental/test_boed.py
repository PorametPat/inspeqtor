import jax.numpy as jnp
from inspeqtor.experimental.boed import (
    _safe_mean_terms,
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
