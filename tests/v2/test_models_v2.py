import jax.numpy as jnp
import inspeqtor.v2.models as ml
from inspeqtor.v2.constant import default_expectation_values_order, X, Y, Z
from inspeqtor.v2.data import get_initial_state, get_observable_operator
from inspeqtor.experimental.physics import calculate_exp
import chex


def test_expectation_values_calculation():
    ideal_expvals = jnp.array(
        [
            calculate_exp(
                jnp.eye(2),
                get_observable_operator(exp.observable),
                get_initial_state(exp.initial_state, dm=True),
            )
            for exp in default_expectation_values_order
        ]
    )

    batch_size = 100
    shape = (batch_size, 2, 2)
    observable = {
        "X": jnp.broadcast_to(X, shape),
        "Y": jnp.broadcast_to(Y, shape),
        "Z": jnp.broadcast_to(Z, shape),
    }

    unitaries = jnp.broadcast_to(jnp.eye(2), shape)

    expvals = ml.shared.get_predict_expectation_value(
        observable,
        unitaries,
        order=default_expectation_values_order,
    )

    assert expvals.shape == (batch_size, len(default_expectation_values_order))

    b = jnp.broadcast_to(ideal_expvals, (batch_size, len(default_expectation_values_order)))

    chex.assert_trees_all_close(b, expvals)
