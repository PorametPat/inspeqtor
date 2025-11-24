import jax.numpy as jnp
from jaxtyping import Array
from inspeqtor.v2.data import ExpectationValue, get_observable_operator


def state_tomography(
    expval: Array, num_qubits: int, order: list[ExpectationValue]
) -> Array:
    operators = jnp.array([get_observable_operator(exp.observable) for exp in order])
    dims = 2**num_qubits
    return (
        jnp.sum(jnp.expand_dims(expval, axis=[-1, -2]) * operators, axis=0)
        + jnp.eye(dims)
    ) / dims
