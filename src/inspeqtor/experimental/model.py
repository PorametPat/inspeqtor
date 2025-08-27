from deprecated import deprecated
import jax
import jax.numpy as jnp
import typing
from flax.typing import VariableDict

from dataclasses import dataclass
import pathlib
import json
from datetime import datetime
from enum import StrEnum
import chex
from numpyro.contrib.module import ParamShape

from .data import ExpectationValue, save_pytree_to_json, load_pytree_from_json, State

from .physics import (
    direct_AFG_estimation,
    direct_AFG_estimation_coefficients,
    to_superop,
    avg_gate_fidelity_from_superop,
    calculate_exp,
)
from .constant import (
    minus_projectors,
    plus_projectors,
    get_default_expectation_values_order,
    X,
    Y,
    Z,
    default_expectation_values_order,
)
from .ctyping import Wos

jax.config.update("jax_enable_x64", True)


def mse(x1: jnp.ndarray, x2: jnp.ndarray):
    return jnp.mean((x1 - x2) ** 2)


def AEF_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, target_unitary: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the absolute error between AGF with respect to given unitary.

    Args:
        y_true (jnp.ndarray): Experimental expectation values
        y_pred (jnp.ndarray): Predicted expectation values
        target_unitary (jnp.ndarray): Target unitary matrix

    Returns:
        jnp.ndarray: loss value
    """
    coefficients = direct_AFG_estimation_coefficients(target_unitary)

    # TODO: Should be squared or absolute value?
    return jnp.abs(
        direct_AFG_estimation(coefficients, y_true)
        - direct_AFG_estimation(coefficients, y_pred)
    )


def WAEE_loss(
    expectation_values_true: jnp.ndarray,
    expectation_values_pred: jnp.ndarray,
    target_unitary: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted absolute error of expectation values

    Args:
        expectation_values_true (jnp.ndarray): Experimental expectation values
        expectation_values_pred (jnp.ndarray): Predicted expectation values
        target_unitary (jnp.ndarray): Unitary used to calculate weight

    Returns:
        jnp.ndarray: Loss value
    """
    coefficients = direct_AFG_estimation_coefficients(target_unitary)
    # The absolute difference between the expectation values
    diff = jnp.abs(expectation_values_true - expectation_values_pred)
    # Weighted by the coefficients
    return jnp.sum(jnp.abs(coefficients) * diff)


def calculate_Pauli_AGF(
    Wos: Wos,
) -> dict[str, jnp.ndarray]:
    """Calculate AGF of Wo with respect to Pauli observable

    Args:
        Wos (Wos): Wos operator

    Returns:
        dict[str, jnp.ndarray]: The AGF in dict form
    """
    AGF_paulis: dict[str, jnp.ndarray] = {}
    # assert isinstance(Wo_params, dict)
    # Calculate the AGF between Wo_model and the Pauli
    # Wos = ensure_wo_type(Wos)
    for pauli_str, pauli_op in zip(["X", "Y", "Z"], [X, Y, Z]):
        # Wo = Wo_2_level_v3(U=Wo_params[pauli_str]["U"], D=Wo_params[pauli_str]["D"])
        # evaluate the fidleity to the Pauli operator
        fidelity = avg_gate_fidelity_from_superop(
            to_superop(Wos[pauli_str]),  # type: ignore
            to_superop(pauli_op),
        )

        AGF_paulis[pauli_str] = fidelity

    return AGF_paulis


def get_predict_expectation_value(
    observable: dict[str, jnp.ndarray],
    unitaries: jnp.ndarray,
    evaluate_expectation_values: list[ExpectationValue],
) -> jnp.ndarray:
    """Calculate expectation values for given evaluate_expectation_values

    Args:
        observable (operators): observable operator
        unitaries (jnp.ndarray): Unitary operators
        evaluate_expectation_values (list[ExpectationValue]): Order of expectation value to be calculated

    Returns:
        jnp.ndarray: Expectation value with order as given with `evaluate_expectation_values`
    """
    predict_expectation_values = jnp.zeros(
        tuple(observable["X"].shape[:-2]) + (len(evaluate_expectation_values),)  # type: ignore
    )

    # Calculate expectation values for all cases
    for idx, exp_case in enumerate(evaluate_expectation_values):
        batch_expectaion_values = calculate_exp(
            unitaries,
            observable[exp_case.observable],  # type: ignore
            exp_case.initial_density_matrix,
        )
        predict_expectation_values = predict_expectation_values.at[..., idx].set(
            batch_expectaion_values
        )

    return predict_expectation_values


def make_get_predict_expectation_value_fn(
    evaluate_expectation_values: list[ExpectationValue],
):
    """Generate a pure function of `get_predict_expectation_value`

    Args:
        evaluate_expectation_values (list[ExpectationValue]): The list of ExpectationValue to make static.
    """

    def _get_predict_expectation_value(
        observable: dict[str, jnp.ndarray],
        unitaries: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate expectation values for given evaluate_expectation_values

        Args:
            observable (operators): observable operator
            unitaries (jnp.ndarray): Unitary operators
            evaluate_expectation_values (list[ExpectationValue]): Order of expectation value to be calculated

        Returns:
            jnp.ndarray: Expectation value with order as given with `evaluate_expectation_values`
        """
        return get_predict_expectation_value(
            observable, unitaries, evaluate_expectation_values
        )

    return _get_predict_expectation_value


class LossMetric(StrEnum):
    MSEE = "MSE[E]"
    AEF = "AE[F]"
    WAEE = "WAE[E]"


def calculate_metric(
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    predicted_expectation_values: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Calculate MSEE, AEF, WAEF at once.

    Args:
        unitaries (jnp.ndarray): Ideal unitary operator
        expectation_values (jnp.ndarray): Experiment expectation value
        predicted_expectation_values (jnp.ndarray): Predicted expectation value

    Returns:
        dict[str, jnp.ndarray]: dict of metrics
    """
    # Calculate the MSE loss
    MSEE = jax.vmap(mse, in_axes=(0, 0))(
        expectation_values, predicted_expectation_values
    )

    # Calculate the AGF loss
    AEF = jax.vmap(AEF_loss, in_axes=(0, 0, 0))(
        expectation_values, predicted_expectation_values, unitaries
    )

    # Calculate WAEE loss
    WAEE = jax.vmap(WAEE_loss, in_axes=(0, 0, 0))(
        expectation_values, predicted_expectation_values, unitaries
    )

    return {
        LossMetric.MSEE: MSEE,
        LossMetric.AEF: AEF,
        LossMetric.WAEE: WAEE,
    }


@deprecated(reason="use ModelData instead")
@dataclass
class ModelState:
    """Dataclass for storing model configurations and parameters."""

    model_config: dict
    model_params: VariableDict

    def save(self, path: pathlib.Path | str):
        """Save model to the given folder

        Args:
            path (pathlib.Path | str): Path to the folder, will be created if not existed.
        """
        # turn the dataclass into a dictionary
        model_params = jax.tree.map(lambda x: x.tolist(), self.model_params)

        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(exist_ok=True)

        with open(path / "model_config.json", "w") as f:
            json.dump(self.model_config, f, indent=4)

        with open(path / "model_params.json", "w") as f:
            json.dump(model_params, f, indent=4)

    @classmethod
    def load(cls, path: pathlib.Path | str):
        """Load model and initialize instance of ModelState from given folder.

        Args:
            path (pathlib.Path | str): Path of folder to be read model data.

        Returns:
            ModelState: Intance of ModelState
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        with open(path / "model_config.json", "r") as f:
            model_config = json.load(f)

        with open(path / "model_params.json", "r") as f:
            model_params = json.load(f)

        # Define is_leaf function to check if a node is a leaf
        # If the node is a leaf, convert it to a jnp.array
        # The function will check if object is list, if list then convert to jnp.array
        def is_leaf(x):
            return isinstance(x, list)

        # Apply the inverse of the tree.map function
        model_params = jax.tree.map(
            lambda x: jnp.array(x), model_params, is_leaf=is_leaf
        )

        return cls(model_config, model_params)


def generate_path_with_datetime(sub_dir: pathlib.Path):
    return sub_dir / datetime.now().strftime("Y%YM%mD%d-H%HM%MS%S")


def hermitian(U: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
    """This is a function that parametrized Hermitian matrix

    Args:
        U (jnp.ndarray): Parameters for unitary operator with shape of (..., 3)
        D (jnp.ndarray): Parameters for diagonal matrix with shape if (..., 2)

    Returns:
        jnp.ndarray: Hermitian matrix of shape (..., 2, 2)
    """
    # parametrize eigenvector matrix being unitary as in https://en.wikipedia.org/wiki/Unitary_matrix

    theta = U[..., 0]
    alpha = U[..., 1]
    beta = U[..., 2]

    lambda_1 = D[..., 0]
    lambda_2 = D[..., 1]

    q_00 = jnp.exp(1j * alpha) * jnp.cos(theta)
    q_01 = jnp.exp(1j * beta) * jnp.sin(theta)
    q_10 = -jnp.exp(-1j * beta) * jnp.sin(theta)
    q_11 = jnp.exp(-1j * alpha) * jnp.cos(theta)

    Q = jnp.zeros(U.shape[:-1] + (2, 2), dtype=jnp.complex_)
    Q = Q.at[..., 0, 0].set(q_00)
    Q = Q.at[..., 0, 1].set(q_01)
    Q = Q.at[..., 1, 0].set(q_10)
    Q = Q.at[..., 1, 1].set(q_11)

    # NOTE: Below is BUG
    # Q_dagger = Q.swapaxes(-2, -1).conj()
    # NOTE: Below is working
    Q_dagger = jnp.swapaxes(Q, -2, -1).conj()

    Diag = jnp.zeros(D.shape[:-1] + (2, 2), dtype=jnp.complex_)
    Diag = Diag.at[..., 0, 0].set(lambda_1)
    Diag = Diag.at[..., 1, 1].set(lambda_2)

    # Return Wos operator
    return jnp.matmul(Q, jnp.matmul(Diag, Q_dagger))


def unitary(params: jnp.ndarray) -> jnp.ndarray:
    """Create unitary matrix from parameters

    Args:
        params (jnp.ndarray): Parameters parametrize the unitary matrix

    Returns:
        jnp.ndarray: Unitary matrix of shape (..., 2, 2)

    """

    theta = params[..., 0]
    alpha = params[..., 1]
    beta = params[..., 2]
    psi = params[..., 3]

    q_00 = jnp.exp(1j * alpha) * jnp.cos(theta)
    q_01 = jnp.exp(1j * beta) * jnp.sin(theta)
    q_10 = -jnp.exp(-1j * beta) * jnp.sin(theta)
    q_11 = jnp.exp(-1j * alpha) * jnp.cos(theta)

    Q = jnp.zeros(params.shape[:-1] + (2, 2), dtype=jnp.complex_)
    Q = Q.at[..., 0, 0].set(q_00)
    Q = Q.at[..., 0, 1].set(q_01)
    Q = Q.at[..., 1, 0].set(q_10)
    Q = Q.at[..., 1, 1].set(q_11)

    psi_ = jnp.expand_dims(jnp.exp(1j * psi / 2), [-2, -1])

    return psi_ * Q


def get_spam(
    params: VariableDict,
) -> tuple[list[ExpectationValue], dict[str, jnp.ndarray]]:
    pair_map = {"+": "-", "-": "+", "0": "1", "1": "0", "r": "l", "l": "r"}
    observables = {"X": X, "Y": Y, "Z": Z}
    for pauli, matrix in observables.items():
        p_10 = params["AM"][pauli]["prob_10"]
        p_01 = params["AM"][pauli]["prob_01"]

        observables[pauli] = (
            matrix
            + (-2 * p_10 * plus_projectors[pauli])
            + (2 * p_01 * minus_projectors[pauli])
        )

    expvals = []
    order_expvals = get_default_expectation_values_order()
    for _expval in order_expvals:
        expval = ExpectationValue(
            initial_state=_expval.initial_state, observable=_expval.observable
        )
        # SP State Preparation error
        SP_correct_prob = params["SP"][_expval.initial_state]
        SP_incorrect_prob = 1 - SP_correct_prob
        expval.initial_density_matrix = SP_correct_prob * State.from_label(
            _expval.initial_state, dm=True
        ) + SP_incorrect_prob * State.from_label(
            pair_map[_expval.initial_state], dm=True
        )
        # AM, And Measurement error
        expval.observable_matrix = observables[_expval.observable]

        expvals.append(expval)

    return expvals, observables


def unitary_to_expvals(output, unitaries: jnp.ndarray) -> jnp.ndarray:
    """Function to be used with Unitary-based model for probabilistic model construction
    with `make_probabilistic_model` function.



    Args:
        output (typing.Any): The output from Unitary-based model
        unitaries (jnp.ndarray): Ideal unitary, ignore in this function.

    Returns:
        jnp.ndarray: Expectation values array
    """
    U = unitary(output)
    return get_predict_expectation_value(
        {
            "X": jnp.broadcast_to(X, shape=(U.shape)),
            "Y": jnp.broadcast_to(Y, shape=(U.shape)),
            "Z": jnp.broadcast_to(Z, shape=(U.shape)),
        },
        U,
        default_expectation_values_order,
    )


def toggling_unitary_to_expvals(
    output: jnp.ndarray, unitaries: jnp.ndarray
) -> jnp.ndarray:
    """Calculate $U_J$ and convert it to expectation values.

    Args:
        output (jnp.ndarray): Parameters parametrized unitary operator
        unitaries (jnp.ndarray): Ideal unitary operators corresponding to the output

    Returns:
        jnp.ndarray: Predicted expectation values
    """
    UJ: jnp.ndarray = unitary(output)  # type: ignore
    UJ_dagger = jnp.swapaxes(UJ, -2, -1).conj()

    X_ = UJ_dagger @ X @ UJ
    Y_ = UJ_dagger @ Y @ UJ
    Z_ = UJ_dagger @ Z @ UJ

    return get_predict_expectation_value(
        {"X": X_, "Y": Y_, "Z": Z_},
        unitaries,
        default_expectation_values_order,
    )


def toggling_unitary_with_spam_to_expvals(
    output, unitaries: jnp.ndarray
) -> jnp.ndarray:
    """To model the SPAM noise with probabilistic model and UJ model

    Note:
        Expected the output structure as follows
        ```python
        spam_params = {
            "SP": {
                "+": jnp.array([0.9]),
                "-": jnp.array([0.9]),
                "r": jnp.array([0.9]),
                "l": jnp.array([0.9]),
                "0": jnp.array([0.9]),
                "1": jnp.array([0.9]),
            },
            "AM": {
                "X": {"prob_10": jnp.array([0.1]), "prob_01": jnp.array([0.1])},
                "Y": {"prob_10": jnp.array([0.1]), "prob_01": jnp.array([0.1])},
                "Z": {"prob_10": jnp.array([0.1]), "prob_01": jnp.array([0.1])},
            },
        }

        output = {
            "model_params": ...,
            "spam_params": spam_params
        }
        ```

    Args:
        output (typing.Any): _description_
        unitaries (jnp.ndarray): _description_

    Returns:
        jnp.ndarray: _description_
    """

    model_output = output["model_params"]
    spam_output = output["spam_params"]

    UJ: jnp.ndarray = unitary(model_output)  # type: ignore
    UJ_dagger = jnp.swapaxes(UJ, -2, -1).conj()

    expectation_value_order, observables = get_spam(spam_output)

    X_ = UJ_dagger @ observables["X"] @ UJ
    Y_ = UJ_dagger @ observables["Y"] @ UJ
    Z_ = UJ_dagger @ observables["Z"] @ UJ

    return get_predict_expectation_value(
        {"X": X_, "Y": Y_, "Z": Z_},
        unitaries,
        expectation_value_order,
    )


def observable_to_expvals(output, unitaries: jnp.ndarray) -> jnp.ndarray:
    """Function to be used with Wo-based model for probabilistic model construction
    with `make_probabilistic_model` function.

    Args:
        output (typing.Any): The output from Wo-based model
        unitaries (jnp.ndarray): Ideal unitary, ignore in this function.

    Returns:
        jnp.ndarray: Expectation values array
    """
    return get_predict_expectation_value(
        observable=output,
        unitaries=unitaries,
        evaluate_expectation_values=default_expectation_values_order,
    )


def model_parse_fn(key, value):
    """This is a parse function to be used with `load_pytree_from_file` function.
    The function will skip parsing config and will return as it is load from file.

    Args:
        key (typing.Any): _description_
        value (typing.Any): _description_

    Returns:
        typing.Any: _description_
    """
    if key == "config":
        return True, value
    elif isinstance(value, list):
        return True, jnp.array(value)
    elif isinstance(value, dict) and len(value) == 1 and "shape" in value:
        return True, ParamShape(shape=tuple(value["shape"]))
    elif isinstance(value, dict):
        return False, None
    else:
        return True, value


@dataclass
class ModelData:
    params: typing.Any
    config: dict[str, typing.Any]

    def to_file(self, path: str | pathlib.Path):
        path = pathlib.Path(path)

        data = {
            "params": self.params,
            "config": self.config,
        }

        save_pytree_to_json(data, path)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> typing.Self:
        data = load_pytree_from_json(path, model_parse_fn)

        return cls(
            params=data["params"],
            config=data["config"],
        )

    def __eq__(self, value):
        if not isinstance(value, type(self)):
            raise ValueError("The compared value is not Model object")

        try:
            chex.assert_trees_all_close(self.params, value.params)
        except AssertionError:
            return False

        return True if value.config == self.config else False
