import jax
import jax.numpy as jnp
import inspeqtor as sq
import chex


def test_check_parity():
    assert sq.utils.check_parity(7) == 1
    assert sq.utils.check_parity(6) == 0


def test_finite_shot_expectation_value():
    key = jax.random.key(0)
    expval = sq.utils.finite_shot_expectation_value(
        key, jnp.array([1.0, 0, 0, 0]), 1000
    )

    assert expval >= -1.0 and expval <= 1.0

    # Check vmap able
    batch_expvals = jax.vmap(
        sq.utils.finite_shot_expectation_value, in_axes=(0, None, None)
    )(jax.random.split(key, 10), jnp.array([1.0, 0, 0, 0]), 1000)
    assert batch_expvals.shape == (10,)


def test_tensor_product():
    state_p = sq.utils.plus_projectors["X"]
    state_m = sq.utils.minus_projectors["X"]
    ideal = jnp.kron(state_p, state_m)

    result = sq.utils.tensor_product(state_p, state_m)

    chex.assert_trees_all_close(ideal, result)

    ideal = jnp.kron(jnp.kron(state_p, state_m), state_p)
    result = sq.utils.tensor_product(state_p, state_m, state_p)

    chex.assert_trees_all_close(ideal, result)


def test_get_measurement_probability():
    expval = sq.data.ExpectationValue("00", "ZZ")
    state = sq.data.get_initial_state(expval.initial_state, dm=True)
    idx_with_one = int(expval.initial_state, base=2)

    results = sq.utils.get_measurement_probability(state, expval.observable)

    # The index is map to bit string: 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
    expected = jnp.zeros(shape=(4,)).at[idx_with_one].set(1.0)
    chex.assert_trees_all_close(results, expected)

    expval = sq.data.ExpectationValue("++", "XY")
    state = sq.data.get_initial_state(expval.initial_state, dm=True)
    results = sq.utils.get_measurement_probability(state, expval.observable)


def test_finite_shot_integration():
    expval = sq.data.ExpectationValue("+1", "XY")
    state = sq.data.get_initial_state(expval.initial_state, dm=True)

    prob = sq.utils.get_measurement_probability(state, expval.observable)

    key = jax.random.key(0)
    expval = sq.utils.finite_shot_expectation_value(key, prob, 1000)

    assert expval >= -1.0 and expval <= 1.0
