import jax.numpy as jnp
import chex
import inspeqtor as sq


def test_expectation_values_calculation():
    ideal_expvals = jnp.array(
        [
            sq.physics.calculate_exp(
                jnp.eye(2),
                sq.data.get_observable_operator(exp.observable),
                sq.data.get_initial_state(exp.initial_state, dm=True),
            )
            for exp in sq.utils.default_expectation_values_order
        ]
    )

    batch_size = 100
    shape = (batch_size, 2, 2)
    observable = {
        "X": jnp.broadcast_to(sq.utils.X, shape),
        "Y": jnp.broadcast_to(sq.utils.Y, shape),
        "Z": jnp.broadcast_to(sq.utils.Z, shape),
    }

    unitaries = jnp.broadcast_to(jnp.eye(2), shape)

    expvals = sq.models.shared.get_predict_expectation_value(
        observable,
        unitaries,
        order=sq.utils.default_expectation_values_order,
    )

    assert expvals.shape == (batch_size, len(sq.utils.default_expectation_values_order))

    b = jnp.broadcast_to(
        ideal_expvals, (batch_size, len(sq.utils.default_expectation_values_order))
    )

    chex.assert_trees_all_close(b, expvals)
