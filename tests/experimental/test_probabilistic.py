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
    batched_matmul,
    get_trace,
    dense_layer,
)
import chex


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


def test_batched_matmul():
    # def batched_matmul(x, w, b):
    #     return jnp.einsum(x, (..., 0), w, (..., 0, 1), (..., 1)) + b

    # 1. Typical model training, control_params.shape == (batch, feature)
    # Expected weights and bias of shape (feature_in, out) and (out)
    # Expected expectation value of shape (batch, out)
    x = jnp.zeros((100, 4))
    w = jnp.zeros((4, 5))
    b = jnp.zeros((5))

    y = batched_matmul(x, w, b)
    assert y.shape == (100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    chex.assert_trees_all_close(x @ w + b, batched_matmul(x, w, b))

    # 2. During BOED

    # 2.1 Model should handle control_params.shape == (extra, design[batch], feature)
    # Model is vecterized by numpyro.plate_stack("plate", control_params.shape[:-1]):
    # Expected weights and bias of shape (extra, design[batch], feature_in, out) and (extra, design[batch], out)
    # Expected expectation value of shape (extra, design[batch], out)
    x = jnp.zeros((10, 100, 4))
    w = jnp.zeros((10, 100, 4, 5))
    b = jnp.zeros((10, 100, 5))

    y = batched_matmul(x, w, b)
    assert y.shape == (10, 100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    def dense(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (4,) and w.shape == (4, 5) and b.shape == (5,)
        return x @ w + b

    expected_y = jax.vmap(jax.vmap(dense, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(
        x, w, b
    )

    chex.assert_trees_all_close(expected_y, batched_matmul(x, w, b))

    # 2.2 Marginal guide handle control_params.shape == (extra, design[batch], feature)
    # Model is vecterized by the design[batch] only
    # Expected weights and bias of shape (design[batch], feature_in, out) and (design[batch], out)
    # Expected expectation value of shape (extra, design[batch], out)
    x = jnp.zeros((10, 100, 4))
    w = jnp.zeros((100, 4, 5))
    b = jnp.zeros((100, 5))

    y = batched_matmul(x, w, b)
    assert y.shape == (10, 100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    expected_y = jax.vmap(jax.vmap(dense, in_axes=(0, 0, 0)), in_axes=(0, None, None))(
        x, w, b
    )

    chex.assert_trees_all_close(expected_y, batched_matmul(x, w, b))


def test_dense_layer():
    # Test normal inference
    for batch_size in [
        (10,),
        (
            20,
            10,
        ),
    ]:
        in_feature = 4
        out_feature = 5
        name = "test_dense"
        trace = get_trace(dense_layer)(
            jnp.ones(batch_size + (in_feature,)), name, in_feature, out_feature
        )

        expected_sites = [
            {
                "name": f"{name}.kernel",
                "type": "sample",
                "value_shape": (4, 5),
                "event_shape": (4, 5),
            },
            {
                "name": f"{name}.bias",
                "type": "sample",
                "value_shape": (5,),
                "event_shape": (5,),
            },
        ]

        for expected_site, site in zip(expected_sites, trace.values(), strict=True):
            assert expected_site["name"] == site["name"]
            assert expected_site["type"] == site["type"]
            assert expected_site["value_shape"] == site["value"].shape
            assert expected_site["event_shape"] == site["fn"].event_shape
