import jax.numpy as jnp
import inspeqtor as sq
import chex


def test_state_from_expectation_values():
    expectation_values_req = sq.data.get_complete_expectation_values(1, states=["0"])

    assert len(expectation_values_req) == 3

    initial_state = sq.data.get_initial_state("0", dm=True)
    expvals = jnp.array(
        [
            sq.physics.calculate_exp(
                jnp.eye(2),
                sq.data.get_observable_operator(exp.observable),
                initial_state,
            )
            for exp in expectation_values_req
        ]
    )

    assert expvals.shape == (len(expectation_values_req),)

    reconstruct_state = sq.physics.state_tomography(expvals, 1, order=expectation_values_req)

    chex.assert_trees_all_close(initial_state, reconstruct_state)

    # Two qubit case
    initial_state_str = "01"
    expectation_values_req = sq.data.get_complete_expectation_values(2)
    expectation_values_req = [
        exp for exp in expectation_values_req if exp.initial_state == initial_state_str
    ]

    assert len(expectation_values_req) == 15

    initial_state = sq.data.get_initial_state(initial_state_str, dm=True)
    expvals = jnp.array(
        [
            sq.physics.calculate_exp(
                jnp.eye(4),
                sq.data.get_observable_operator(exp.observable),
                initial_state,
            )
            for exp in expectation_values_req
        ]
    )

    assert expvals.shape == (len(expectation_values_req),)

    reconstruct_state = sq.physics.state_tomography(expvals, 2, order=expectation_values_req)

    chex.assert_trees_all_close(initial_state, reconstruct_state)
