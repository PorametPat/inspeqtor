import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from inspeqtor.experimental.probabilistic import (
    eigenvalue_to_binary,
    binary_to_eigenvalue,
    expectation_value_to_eigenvalue,
    construct_normal_prior_from_samples,
    get_args_of_distribution,
    ProbabilisticModel,
)


def test_eigenvalue_binary_conversion():
    import chex

    r = eigenvalue_to_binary(jnp.array([-1, 1]))

    chex.assert_trees_all_close(r, jnp.array([1, 0]))

    r = binary_to_eigenvalue(jnp.array([0, 1]))

    chex.assert_trees_all_close(r, jnp.array([1, -1]))


def test_expectation_value_to_eigenvalue():
    import chex

    result = expectation_value_to_eigenvalue(
        jnp.array([[1, -1, 0], [-1, 0, 1]]), SHOTS=10
    )

    expect_result = jnp.array(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]
    )

    assert jnp.array_equal(result, expect_result)

    # Hard core test
    expvals = jnp.linspace(-1, 1, 1001)
    result = expectation_value_to_eigenvalue(expvals, SHOTS=3000)

    chex.assert_trees_all_close(expvals, result.mean(axis=-1))


@pytest.mark.skip("Not finished")
def test_save_and_load_model(guide, svi_params, shots, hidden_sizes):
    posterior_samples = guide.sample_posterior(
        jax.random.key(0), svi_params, sample_shape=(10000,)
    )
    posterior = construct_normal_prior_from_samples(posterior_samples)
    posterior = {
        name.split("/")[1]: prior
        for name, prior in posterior.items()
        if name.startswith("nn/")
    }

    posterior = jax.tree.map(
        get_args_of_distribution,
        posterior,
        is_leaf=lambda x: isinstance(x, dist.Distribution),
    )

    probabilistic_model = ProbabilisticModel(
        posterior=posterior,
        shots=shots,
        hidden_sizes=hidden_sizes,
    )

    probabilistic_model.to_file("./test/0001/model.json")

    ProbabilisticModel.from_file("./test/0001/model.json")
