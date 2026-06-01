import jax.numpy as jnp
from inspeqtor.v2.data import ExpectationValue, get_observable_operator


def state_tomography(
    expval: jnp.ndarray, num_qubits: int, order: list[ExpectationValue]
) -> jnp.ndarray:
    operators = jnp.ndarray([get_observable_operator(exp.observable) for exp in order])
    dims = 2**num_qubits
    return (
        jnp.sum(jnp.expand_dims(expval, axis=[-1, -2]) * operators, axis=0)
        + jnp.eye(dims)
    ) / dims


def U1(theta: jnp.ndarray) -> jnp.ndarray:
    # https://github.com/Qiskit/qiskit/blob/stable/2.4/qiskit/circuit/library/standard_gates/u1.py#L24-L170
    return jnp.ndarray([[1, 0], [0, jnp.exp(1j * theta)]])


def U2(phi: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    # https://github.com/Qiskit/qiskit/blob/stable/2.4/qiskit/circuit/library/standard_gates/u2.py#L23-L148
    return (
        1
        / jnp.sqrt(2)
        * jnp.ndarray(
            [
                [1, -jnp.exp(1j * lam)],
                [jnp.exp(1j * phi), jnp.exp(1j * (phi + lam))],
            ]
        )
    )


def U3(theta: jnp.ndarray, phi: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    # https://github.com/Qiskit/qiskit/blob/stable/2.4/qiskit/circuit/library/standard_gates/u3.py#L27-L193

    return jnp.ndarray(
        [
            [jnp.cos(theta / 2), -jnp.exp(1j * lam) * jnp.sin(theta / 2)],
            [
                jnp.exp(1j * phi) * jnp.sin(theta / 2),
                jnp.exp(1j * (phi + lam)) * jnp.cos(theta / 2),
            ],
        ]
    )
