import jax
import jax.numpy as jnp
import typing
from flax import linen as nn
from flax.typing import VariableDict

from dataclasses import dataclass, asdict
import pathlib
import json
from datetime import datetime
from enum import StrEnum
import pandas as pd


from .data import ExpectationValue
from .constant import default_expectation_values_order
from .physics import (
    direct_AFG_estimation,
    direct_AFG_estimation_coefficients,
    to_superop,
    avg_gate_fidelity_from_superop,
    calculate_exp,
    X,
    Y,
    Z,
)
from .pulse import PulseSequence
from .typing import WoParams, ensure_wo_params_type


def Wo_2_level_v3(U: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
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

    Q = jnp.zeros(U.shape[:-1] + (2, 2), dtype=jnp.complex64)
    Q = Q.at[..., 0, 0].set(q_00)
    Q = Q.at[..., 0, 1].set(q_01)
    Q = Q.at[..., 1, 0].set(q_10)
    Q = Q.at[..., 1, 1].set(q_11)

    # NOTE: Below is BUG
    # Q_dagger = Q.swapaxes(-2, -1).conj()
    # NOTE: Below is working
    Q_dagger = jnp.swapaxes(Q, -2, -1).conj()

    Diag = jnp.zeros(D.shape[:-1] + (2, 2), dtype=jnp.complex64)
    Diag = Diag.at[..., 0, 0].set(lambda_1)
    Diag = Diag.at[..., 1, 1].set(lambda_2)

    # Return Wos operator
    return jnp.matmul(Q, jnp.matmul(Diag, Q_dagger))


class BasicBlackBox(nn.Module):
    feature_size: int
    hidden_sizes_1: typing.Sequence[int] = (20, 10)
    hidden_sizes_2: typing.Sequence[int] = (20, 10)
    pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(features=self.feature_size)(x)
        # Apply a activation function
        x = nn.relu(x)
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # ! Hotfix for the typing complaint
            # ! That _x might be unbound. But it is actually bound.
            _x = jnp.zeros_like(x)
            # Sub hidden layer
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


class BasicBlackBoxV2(nn.Module):
    hidden_sizes_1: typing.Sequence[int] = (20, 10)
    hidden_sizes_2: typing.Sequence[int] = (20, 10)
    pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Apply a dense layer for each hidden size
        for hidden_size in self.hidden_sizes_1:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            # Copy the input
            _x = jnp.copy(x)
            for hidden_size in self.hidden_sizes_2:
                _x = nn.Dense(features=hidden_size)(_x)
                _x = nn.relu(_x)

            Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS)(_x)
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS)(_x)
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


def mse(x1: jnp.ndarray, x2: jnp.ndarray):
    return jnp.mean((x1 - x2) ** 2)


def AEF_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, target_unitary: jnp.ndarray
) -> jnp.ndarray:
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
):
    coefficients = direct_AFG_estimation_coefficients(target_unitary)
    # The absolute difference between the expectation values
    diff = jnp.abs(expectation_values_true - expectation_values_pred)
    # Weighted by the coefficients
    return jnp.sum(jnp.abs(coefficients) * diff)


def calculate_Pauli_AGF(Wo_params: WoParams) -> dict[str, jnp.ndarray]:
    AGF_paulis: dict[str, jnp.ndarray] = {}
    assert isinstance(Wo_params, dict)
    # Calculate the AGF between Wo_model and the Pauli
    for pauli_str, pauli_op in zip(["X", "Y", "Z"], [X, Y, Z]):
        Wo = Wo_2_level_v3(U=Wo_params[pauli_str]["U"], D=Wo_params[pauli_str]["D"])
        # evaluate the fidleity to the Pauli operator
        fidelity = avg_gate_fidelity_from_superop(to_superop(Wo), to_superop(pauli_op))

        AGF_paulis[pauli_str] = fidelity

    return AGF_paulis


def get_predict_expectation_value(
    Wos_params: WoParams,
    unitaries: jnp.ndarray,
    evaluate_expectation_values: list[ExpectationValue],
) -> jnp.ndarray:
    predict_expectation_values = jnp.zeros(
        tuple(unitaries.shape[:-2]) + (len(evaluate_expectation_values),)
    )

    Wos_params = ensure_wo_params_type(Wos_params)

    # Calculate expectation values for all cases
    for idx, exp_case in enumerate(evaluate_expectation_values):
        Wo = Wo_2_level_v3(
            Wos_params[exp_case.observable]["U"], Wos_params[exp_case.observable]["D"]
        )

        batch_expectaion_values = calculate_exp(
            unitaries,
            Wo,
            exp_case.initial_density_matrix,
        )
        predict_expectation_values = predict_expectation_values.at[..., idx].set(
            batch_expectaion_values
        )

    return predict_expectation_values


class LossMetric(StrEnum):
    MSEE = "MSE[E]"
    AEF = "AE[F]"
    WAEE = "WAE[E]"


def calculate_metrics(
    # The model to be used for prediction
    model: nn.Module,
    model_params: VariableDict,
    # Input data to the model
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    # Experimental data
    expectation_values: jnp.ndarray,
    # model keyword arguments
    model_kwargs: dict = {},
):
    """To Calculate the metrics of the model
    1. MSE Loss between the predicted expectation values and the experimental expectation values
    2. Average Gate Fidelity between the Pauli matrices to the Wo_model matrices
    3. AGF Loss between the prediction from model and the experimental expectation values

    Args:
        model (sq.model.nn.Module): The model to be used for prediction
        model_params (sq.model.VariableDict): The model parameters
        pulse_parameters (jnp.ndarray): The pulse parameters
        unitaries (jnp.ndarray): Ideal unitaries
        expectation_values (jnp.ndarray): Experimental expectation values
        model_kwargs (dict): Model keyword arguments
    """

    # Calculate Wo_params
    Wo_params = model.apply(model_params, pulse_parameters, **model_kwargs)

    # Calculate the predicted expectation values using model
    predicted_expvals = get_predict_expectation_value(
        Wo_params,
        unitaries,
        default_expectation_values_order,
    )

    # Calculate the metrics
    metrics = calculate_metric(
        unitaries=unitaries,
        expectation_values=expectation_values,
        predicted_expectation_values=predicted_expvals,
    )

    AGF_paulis = jax.vmap(calculate_Pauli_AGF, in_axes=(0))(
        Wo_params,
    )

    return {
        **metrics,
        "AGF_Paulis": AGF_paulis,
    }


def calculate_metric(
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    predicted_expectation_values: jnp.ndarray,
):
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


def loss_fn(
    params: VariableDict,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    model: nn.Module,
    loss_metric: LossMetric,
    model_kwargs: dict = {},
):
    # Calculate the metrics
    metrics = calculate_metrics(
        model=model,
        model_params=params,
        pulse_parameters=pulse_parameters,
        unitaries=unitaries,
        expectation_values=expectation_values,
        model_kwargs=model_kwargs,
    )

    # Take mean of all the metrics
    metrics = jax.tree.map(jnp.mean, metrics)

    # ! Grab the metric in the `metrics`
    loss = metrics[loss_metric]

    return (loss, metrics)


@dataclass
class ModelState:
    model_config: dict
    model_params: VariableDict

    def save(self, path: pathlib.Path | str):
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


@dataclass
class DataConfig:
    EXPERIMENT_IDENTIFIER: str
    hamiltonian: str
    pulse_sequence: dict

    def to_file(self, path: typing.Union[str, pathlib.Path]):
        if isinstance(path, str):
            path = pathlib.Path(path)

        path.mkdir(exist_ok=True)
        with open(f"{path}/data_config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_file(cls, path: typing.Union[str, pathlib.Path]):
        if isinstance(path, str):
            path = pathlib.Path(path)
        with open(path / "data_config.json", "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def generate_path_with_datetime(sub_dir: pathlib.Path):
    return sub_dir / datetime.now().strftime("Y%YM%mD%d-H%HM%MS%S")


def save_model(
    path: pathlib.Path | str,
    experiment_identifier: str,
    pulse_sequence: PulseSequence,
    hamiltonian: typing.Union[str, typing.Callable],
    model_config: dict,
    model_params: VariableDict,
    history: typing.Sequence[dict[typing.Any, typing.Any]] | None = None,
    with_auto_datetime: bool = True,
):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    path = generate_path_with_datetime(path) if with_auto_datetime else path

    path.mkdir(parents=True, exist_ok=True)

    model_state = ModelState(
        model_config=model_config,
        model_params=model_params,
    )

    model_state.save(path / "model_state")

    if history is not None:
        # Save the history
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(path / "history.csv", index=False)

    # Save the data config
    data_config = DataConfig(
        EXPERIMENT_IDENTIFIER=experiment_identifier,
        hamiltonian=(
            hamiltonian if isinstance(hamiltonian, str) else hamiltonian.__name__
        ),
        pulse_sequence=pulse_sequence.to_dict(),
    )

    data_config.to_file(path)

    return path


def load_model(path: pathlib.Path | str, skip_history: bool = False):
    if isinstance(path, str):
        path = pathlib.Path(path)
    model_state = ModelState.load(path / "model_state")
    if skip_history:
        hist_df = None
    else:
        hist_df = pd.read_csv(path / "history.csv").to_dict(orient="records")
    data_config = DataConfig.from_file(path)

    return (
        model_state,
        hist_df,
        data_config,
    )
