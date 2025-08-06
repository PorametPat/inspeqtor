from deprecated import deprecated
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
import chex
from numpyro.contrib.module import ParamShape

from .data import ExpectationValue, save_pytree_to_json, load_pytree_from_json
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
from .control import ControlSequence
from .typing import Wos

jax.config.update("jax_enable_x64", True)


def Wo_2_level_v3(U: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
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
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS, name=f"U_{op}")(
                _x
            )
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}")(
                _x
            )
            # Apply the activation function
            diag_params = nn.tanh(diag_params)

            Wos_params[op] = {
                "U": unitary_params,
                "D": diag_params,
            }

        return Wos_params


def make_basic_blackbox_model(
    unitary_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: 2
    * jnp.pi
    * nn.hard_sigmoid(x),
    diagonal_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: (
        2 * nn.hard_sigmoid(x)
    )
    - 1,
) -> type[nn.Module]:
    """Function to create Blackbox constructor with custom activation functions for unitary and diagonal output

    Args:
        unitary_activation_fn (_type_, optional): Activation function for unitary parameters. Defaults to lambdax:2*jnp.pi*nn.hard_sigmoid(x).
        diagonal_activation_fn (_type_, optional): Activation function for diagonal parameters. Defaults to lambdax:(2 * nn.hard_sigmoid(x))-1.

    Returns:
        type[nn.Module]: Constructor of the Blackbox model
    """

    class BlackBox(nn.Module):
        hidden_sizes_1: typing.Sequence[int] = (20, 10)
        hidden_sizes_2: typing.Sequence[int] = (20, 10)
        pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

        NUM_UNITARY_PARAMS: int = 3
        NUM_DIAGONAL_PARAMS: int = 2

        _unitary_activation_fn: typing.Callable = unitary_activation_fn
        _diagonal_activation_fn: typing.Callable = diagonal_activation_fn

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            # Apply a dense layer for each hidden size
            for hidden_size in self.hidden_sizes_1:
                x = nn.Dense(features=hidden_size)(x)
                x = nn.relu(x)

            # Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
            Wos: dict[str, jnp.ndarray] = dict()
            for op in self.pauli_operators:
                # Sub hidden layer
                # Copy the input
                _x = jnp.copy(x)
                for hidden_size in self.hidden_sizes_2:
                    _x = nn.Dense(features=hidden_size)(_x)
                    _x = nn.relu(_x)

                # Wos_params[op] = dict()
                # For the unitary part, we use a dense layer with 3 features
                unitary_params = nn.Dense(
                    features=self.NUM_UNITARY_PARAMS, name=f"U_{op}"
                )(_x)
                # Apply sigmoid to this layer
                unitary_params = self._unitary_activation_fn(unitary_params)
                # For the diagonal part, we use a dense layer with 1 feature
                diag_params = nn.Dense(
                    features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}"
                )(_x)
                # Apply the activation function
                diag_params = self._diagonal_activation_fn(diag_params)

                Wos[op] = Wo_2_level_v3(unitary_params, diag_params)

            return Wos

    return BlackBox


class BasicBlackBoxV3(nn.Module):
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
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS, name=f"U_{op}")(
                _x
            )
            # Apply sigmoid to this layer
            unitary_params = 2 * jnp.pi * nn.hard_sigmoid(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}")(
                _x
            )
            # Apply the activation function
            diag_params = (2 * nn.hard_sigmoid(diag_params)) - 1

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
    Wos: Wos,
    unitaries: jnp.ndarray,
    evaluate_expectation_values: list[ExpectationValue],
) -> jnp.ndarray:
    """Calculate expectation values for given evaluate_expectation_values

    Args:
        Wos (Wos): Wos operator
        unitaries (jnp.ndarray): Unitary operators
        evaluate_expectation_values (list[ExpectationValue]): Order of expectation value to be calculated

    Returns:
        jnp.ndarray: Expectation value with order as given with `evaluate_expectation_values`
    """
    # predict_expectation_values = jnp.zeros(
    #     tuple(unitaries.shape[:-2]) + (len(evaluate_expectation_values),)
    # )
    predict_expectation_values = jnp.zeros(
        tuple(Wos["X"].shape[:-2]) + (len(evaluate_expectation_values),)  # type: ignore
    )

    # Calculate expectation values for all cases
    for idx, exp_case in enumerate(evaluate_expectation_values):
        batch_expectaion_values = calculate_exp(
            unitaries,
            Wos[exp_case.observable],  # type: ignore
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


def loss_fn(
    params: VariableDict,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    model: nn.Module,
    loss_metric: LossMetric,
    model_kwargs: dict = {},
    calculate_metrics_fn: typing.Callable = calculate_metrics,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Calculate losses and return the specified one as the first element in the tuple

    Args:
        params (VariableDict): Model parameters
        pulse_parameters (jnp.ndarray): Control parameters
        unitaries (jnp.ndarray): Ideal unitary
        expectation_values (jnp.ndarray): Experimental expectation value
        model (nn.Module): Model instance
        loss_metric (LossMetric): The choice of loss to be optimized
        model_kwargs (dict, optional): Keyword arguments for the model. Defaults to {}.
        calculate_metrics_fn:

    Returns:
        tuple[jnp.ndarray, dict[str, jnp.ndarray]]: loss, and all metrices
    """
    # Calculate the metrics
    metrics = calculate_metrics_fn(
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


@deprecated("Prefer explicitly specify hamiltonian to use.")
@dataclass
class DataConfig:
    """Config for pulse sequence and Hamiltonian used for Whitebox"""

    EXPERIMENT_IDENTIFIER: str
    hamiltonian: str
    control_sequence: dict

    def to_file(self, path: typing.Union[str, pathlib.Path]):
        """Save the data config to file.

        Args:
            path (typing.Union[str, pathlib.Path]): Path to the folder, will be created if not existed.
        """
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
        """Load model and initialize instance of DataConfig from given folder.

        Args:
            path (typing.Union[str, pathlib.Path]): Path to the folder, will be created if not existed.

        Returns:
            DataConfig: Intance of DataConfig read from folder.
        """
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
    control_sequence: ControlSequence,
    hamiltonian: typing.Union[str, typing.Callable],
    model_config: dict,
    model_params: VariableDict,
    history: typing.Sequence[dict[typing.Any, typing.Any]] | None = None,
    with_auto_datetime: bool = True,
) -> pathlib.Path:
    """Function to save training result, including model config, training history, data config.

    Args:
        path (pathlib.Path | str): Path to folder to save data
        experiment_identifier (str): Experiment identifier
        control_sequence (PulseSequence): Pulse sequence used in the training
        hamiltonian (typing.Union[str, typing.Callable]): Ideal Hamiltonian used for Whitebox
        model_config (dict): Configuration of model, used for model initialization
        model_params (VariableDict): Parameters of model.
        history (typing.Sequence[dict[typing.Any, typing.Any]] | None, optional): Training history. Defaults to None.
        with_auto_datetime (bool, optional): True to automatically append datetime as a parent folder. Defaults to True.

    Returns:
        pathlib.Path: Path to the saved data.
    """
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
        control_sequence=control_sequence.to_dict(),
    )

    data_config.to_file(path)

    return path


def load_model(path: pathlib.Path | str, skip_history: bool = False):
    """Load model state, training history, data config from given folder path

    Args:
        path (pathlib.Path | str): Path to the folder.
        skip_history (bool, optional): True to skip reading the history. Defaults to False.

    Returns:
        (ModelState, pd.DataFrame, DataConfig): tuple of Intance of `ModelState`, `pd.DataFrame`, and `DataConfig`.
    """
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


class UnitaryModel(nn.Module):
    # feature_size: int
    hidden_sizes: list[int]

    NUM_UNITARY_PARAMS: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Apply a dense layer for each hidden size
        # x = nn.Dense(features=self.feature_size)(x)
        # x = nn.relu(x)

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        # For the unitary part, we use a dense layer with 3 features
        x = nn.Dense(features=self.NUM_UNITARY_PARAMS)(x)
        # Apply sigmoid to this layer
        x = 2 * jnp.pi * nn.hard_sigmoid(x)

        return x


def calculate_metrics_v2(
    # The model to be used for prediction
    model: UnitaryModel,
    model_params: VariableDict,
    # Input data to the model
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    # Experimental data
    expectation_values: jnp.ndarray,
    # model keyword arguments
    model_kwargs: dict = {},
):
    """Caculate for unitary-based Blackbox model

    Args:
        model (sq.model.nn.Module): The model to be used for prediction
        model_params (sq.model.VariableDict): The model parameters
        pulse_parameters (jnp.ndarray): The pulse parameters
        unitaries (jnp.ndarray): Ideal unitaries
        expectation_values (jnp.ndarray): Experimental expectation values
        model_kwargs (dict): Model keyword arguments

    Returns:
        _type_: _description_
    """

    # Predict Unitary parameters
    unitary_params = model.apply(model_params, pulse_parameters, **model_kwargs)

    U = unitary(unitary_params)  # type: ignore

    predicted_expvals = get_predict_expectation_value(
        {"X": X, "Y": Y, "Z": Z},
        U,
        default_expectation_values_order,
    )

    # Calculate the metrics
    metrics = calculate_metric(
        unitaries=unitaries,
        expectation_values=expectation_values,
        predicted_expectation_values=predicted_expvals,
    )

    return metrics


def construct_unitary_model_from_config(
    config: dict[str, int],
) -> tuple[nn.Module, dict[str, int | list[int]]]:
    """Construct Unitary-based model from the config

    Args:
        config (dict[str, int]): Config of the model

    Returns:
        tuple[nn.Module, dict[str, int | list[int]]]: Unitary-based model
    """

    model_config: dict[str, int | list[int]] = {
        "hidden_sizes": [config["hidden_size_1"], config["hidden_size_2"]],
    }

    return (
        UnitaryModel(
            hidden_sizes=[config["hidden_size_1"], config["hidden_size_2"]],
        ),
        model_config,
    )


def construct_wo_model_from_config(
    config: dict[str, int], model_constructor: type[nn.Module]
):
    """Construct Wo-based model from config

    Args:
        config (dict[str, int]): Config of the model
        model_constructor (type[nn.Module]): Model constructor of the Wo-based model

    Returns:
        tuple[nn.Module, dict[str, int | list[int]]]: Wo-based model
    """
    HIDDEN_LAYER_1_1 = config["hidden_layer_1_1"]
    HIDDEN_LAYER_1_2 = config["hidden_layer_1_2"]
    HIDDEN_LAYER_2_1 = config["hidden_layer_2_1"]
    HIDDEN_LAYER_2_2 = config["hidden_layer_2_2"]

    HIDDEN_LAYER_1 = [i for i in [HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2] if i != 0]
    HIDDEN_LAYER_2 = [i for i in [HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2] if i != 0]

    model_config = {
        "hidden_sizes_1": HIDDEN_LAYER_1,
        "hidden_sizes_2": HIDDEN_LAYER_2,
    }

    model = model_constructor(**model_config)
    return model, model_config


def make_dropout_blackbox_model(
    unitary_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: 2
    * jnp.pi
    * nn.hard_sigmoid(x),
    diagonal_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: (
        2 * nn.hard_sigmoid(x)
    )
    - 1,
) -> type[nn.Module]:
    """Function to create Blackbox constructor with custom activation functions for unitary and diagonal output

    Args:
        unitary_activation_fn (_type_, optional): Activation function for unitary parameters. Defaults to lambdax:2*jnp.pi*nn.hard_sigmoid(x).
        diagonal_activation_fn (_type_, optional): Activation function for diagonal parameters. Defaults to lambdax:(2 * nn.hard_sigmoid(x))-1.

    Returns:
        type[nn.Module]: Constructor of the Blackbox model
    """

    class BlackBox(nn.Module):
        dropout_rate: float = 0.2
        hidden_sizes_1: typing.Sequence[int] = (20, 10)
        hidden_sizes_2: typing.Sequence[int] = (20, 10)
        pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

        NUM_UNITARY_PARAMS: int = 3
        NUM_DIAGONAL_PARAMS: int = 2

        _unitary_activation_fn: typing.Callable = unitary_activation_fn
        _diagonal_activation_fn: typing.Callable = diagonal_activation_fn

        @nn.compact
        def __call__(
            self, x: jnp.ndarray, stochastic: bool = True
        ) -> dict[str, jnp.ndarray]:
            # Apply a dense layer for each hidden size
            for hidden_size in self.hidden_sizes_1:
                x = nn.Dense(features=hidden_size)(x)
                x = nn.relu(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not stochastic)(x)

            # Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
            Wos: dict[str, jnp.ndarray] = dict()
            for op in self.pauli_operators:
                # Sub hidden layer
                # Copy the input
                _x = jnp.copy(x)
                for hidden_size in self.hidden_sizes_2:
                    _x = nn.Dense(features=hidden_size)(_x)
                    _x = nn.relu(_x)
                    _x = nn.Dropout(
                        rate=self.dropout_rate, deterministic=not stochastic
                    )(_x)

                # For the unitary part, we use a dense layer with 3 features
                unitary_params = nn.Dense(
                    features=self.NUM_UNITARY_PARAMS, name=f"U_{op}"
                )(_x)
                # Apply sigmoid to this layer
                unitary_params = self._unitary_activation_fn(unitary_params)
                # For the diagonal part, we use a dense layer with 1 feature
                diag_params = nn.Dense(
                    features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}"
                )(_x)
                # Apply the activation function
                diag_params = self._diagonal_activation_fn(diag_params)

                Wos[op] = Wo_2_level_v3(unitary_params, diag_params)

            return Wos

    return BlackBox


def model_parse_fn(key, value):
    """This is a parse function to be used with `load_pytree_from_file` function.
    The function will skip parsing config and will return as it is load from file.

    Args:
        key (_type_): _description_
        value (_type_): _description_

    Returns:
        _type_: _description_
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
    params: chex.ArrayTree
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


class StatisticalModel(ModelData):
    pass
