import jax.numpy as jnp
from inspeqtor.v2.data import ExpectationValue, get_initial_state
from inspeqtor.experimental.physics import calculate_exp


def get_predict_expectation_value(
    observable: dict[str, jnp.ndarray],
    unitaries: jnp.ndarray,
    order: list[ExpectationValue],
) -> jnp.ndarray:
    """Calculate expectation values for given order

    Args:
        observable (operators): observable operator
        unitaries (jnp.ndarray): Unitary operators
        order (list[ExpectationValue]): Order of expectation value to be calculated

    Returns:
        jnp.ndarray: Expectation value with order as given with `order`
    """

    return jnp.array(
        [
            calculate_exp(
                unitaries,
                observable[exp.observable],
                get_initial_state(exp.initial_state, dm=True),
            )
            for exp in order
        ]
    ).transpose()

