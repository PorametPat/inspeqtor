import jax.numpy as jnp
import numpy as np

from ..experimental.physics import check_hermitian, X, Y, Z
from ..experimental.data import ExpectationValue

from forest.benchmarking.observable_estimation import (  # type: ignore
    ExperimentResult,
    ExperimentSetting,
    TensorProductState,
    PauliTerm,
)
from forest.benchmarking.tomography import (  # type: ignore
    linear_inv_process_estimate,
    pgdb_process_estimate,
)
from forest.benchmarking.operator_tools.superoperator_transformations import (  # type: ignore
    choi2superop,
)


initial_state_mapping = {
    "0": "Z+",
    "1": "Z-",
    "+": "X+",
    "-": "X-",
    "r": "Y+",
    "l": "Y-",
}


def process_tomography(
    rho_0: jnp.ndarray,
    rho_1: jnp.ndarray,
    rho_p: jnp.ndarray,
    rho_m: jnp.ndarray,
):
    Lambda = (1 / 2) * jnp.block([[jnp.eye(2), X], [X, -jnp.eye(2)]])

    rho_p1 = rho_0
    rho_p4 = rho_1
    rho_p2 = rho_p - 1j * rho_m - (1 - 1j) * (rho_p1 + rho_p4) / 2
    rho_p3 = rho_p + 1j * rho_m - (1 + 1j) * (rho_p1 + rho_p4) / 2

    # NOTE: This is difference from the block 8.5 in Nielsen and Chuang.
    Rho = jnp.block([[rho_p1, rho_p3], [rho_p2, rho_p4]])
    # Rho = jnp.block([[rho_p1, rho_p2], [rho_p3, rho_p4]])

    # return choi2superop(Rho)

    Chi = Lambda @ Rho @ Lambda

    # Check hermitain
    check_hermitian(Chi)

    # From process matrix to choi matrix
    P_0 = jnp.eye(2)
    P_1 = X
    # P_2 = -1*Y
    P_2 = Y
    P_3 = Z

    Ps = [P_0, P_1, P_2, P_3]
    choi_matrix = jnp.zeros((4, 4))
    for m, pauli_m in enumerate(Ps):
        for n, pauli_n in enumerate(Ps):
            superket_m = pauli_m.flatten().reshape(-1, 1)
            superbra_n = pauli_n.flatten().conj()
            # Outer product
            choi_basis = Chi[m, n] * jnp.outer(superket_m, superbra_n)
            choi_matrix += choi_basis

    return jnp.array(choi2superop(np.array(choi_matrix)))


def forest_process_tomography(
    expvals: list[ExpectationValue],
    is_linear_inv: bool = True,
):
    """Calculate the process tomography estimate using the Forest-benchmarking library.

    Args:
        expvals (list[sq.data.ExpectationValue]): The list of expectation values.

    Returns:
        _type_: The process tomography estimate in the Choi matrix representation.
    """
    # Extract results
    exp_results = []

    for expval in expvals:
        assert isinstance(expval.expectation_value, float), (
            "Expectation value is not float"
        )

        exp_results.append(
            ExperimentResult(
                setting=ExperimentSetting(
                    observable=PauliTerm.from_list(
                        [(expval.observable, 0)]
                    ),  # NOTE: Assumes qubit 0
                    in_state=TensorProductState.from_str(
                        f"{initial_state_mapping[expval.initial_state]}_0"  # NOTE: Assumes qubit 0
                    ),
                ),
                expectation=expval.expectation_value,
                total_counts=1000,  # TODO: Should be set to the actual number of counts?
            )
        )

    if is_linear_inv:
        est = linear_inv_process_estimate(exp_results, [0])  # NOTE: Assumes qubit 0
    else:
        est = pgdb_process_estimate(exp_results, [0])  # NOTE: Assumes qubit 0

    return est
