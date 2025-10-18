import deprecated
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import typing
import jaxtyping
from flax import linen as nn
from flax.typing import VariableDict

from ..model import (
    LossMetric,
    calculate_metric,
    hermitian,
    observable_to_expvals,
    unitary_to_expvals,
    toggling_unitary_to_expvals,
    toggling_unitary_with_spam_to_expvals,
)

from ..optimize import DataBundled, HistoryEntryV3
from ..utils import dataloader


class WoModel(nn.Module):
    shared_layers: typing.Sequence[int] = (20, 10)
    pauli_layers: typing.Sequence[int] = (20, 10)
    pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    unitary_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = (
        lambda x: 2 * jnp.pi * nn.hard_sigmoid(x)
    )
    diagonal_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = (
        lambda x: (2 * nn.hard_sigmoid(x)) - 1
    )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        # Apply a dense layer for each hidden size
        for hidden_size in self.shared_layers:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        # Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        Wos: dict[str, jnp.ndarray] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            # Copy the input
            _x = jnp.copy(x)
            for hidden_size in self.pauli_layers:
                _x = nn.Dense(features=hidden_size)(_x)
                _x = nn.relu(_x)

            # Wos_params[op] = dict()
            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS, name=f"U_{op}")(
                _x
            )
            # Apply sigmoid to this layer
            unitary_params = self.unitary_activation_fn(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}")(
                _x
            )
            # Apply the activation function
            diag_params = self.diagonal_activation_fn(diag_params)

            Wos[op] = hermitian(unitary_params, diag_params)

        return Wos


class UnitaryModel(nn.Module):
    # feature_size: int
    hidden_sizes: list[int]

    NUM_UNITARY_PARAMS: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Apply a dense layer for each hidden size

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)

        # For the unitary part, we use a dense layer with 3 features
        x = nn.Dense(features=self.NUM_UNITARY_PARAMS)(x)
        # Apply sigmoid to this layer
        x = 2 * jnp.pi * nn.hard_sigmoid(x)

        return x


class WoDropoutModel(nn.Module):
    dropout_rate: float = 0.2
    shared_layers: typing.Sequence[int] = (20, 10)
    shared_layers: typing.Sequence[int] = (20, 10)
    pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

    NUM_UNITARY_PARAMS: int = 3
    NUM_DIAGONAL_PARAMS: int = 2

    _unitary_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = (
        lambda x: 2 * jnp.pi * nn.hard_sigmoid(x)
    )
    _diagonal_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = (
        lambda x: (2 * nn.hard_sigmoid(x)) - 1
    )

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, stochastic: bool = True
    ) -> dict[str, jnp.ndarray]:
        # Apply a dense layer for each hidden size
        for hidden_size in self.shared_layers:
            x = nn.Dense(features=hidden_size)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not stochastic)(x)

        # Wos_params: dict[str, dict[str, jnp.ndarray]] = dict()
        Wos: dict[str, jnp.ndarray] = dict()
        for op in self.pauli_operators:
            # Sub hidden layer
            # Copy the input
            _x = jnp.copy(x)
            for hidden_size in self.shared_layers:
                _x = nn.Dense(features=hidden_size)(_x)
                _x = nn.relu(_x)
                _x = nn.Dropout(rate=self.dropout_rate, deterministic=not stochastic)(
                    _x
                )

            # For the unitary part, we use a dense layer with 3 features
            unitary_params = nn.Dense(features=self.NUM_UNITARY_PARAMS, name=f"U_{op}")(
                _x
            )
            # Apply sigmoid to this layer
            unitary_params = self._unitary_activation_fn(unitary_params)
            # For the diagonal part, we use a dense layer with 1 feature
            diag_params = nn.Dense(features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}")(
                _x
            )
            # Apply the activation function
            diag_params = self._diagonal_activation_fn(diag_params)

            Wos[op] = hermitian(unitary_params, diag_params)

        return Wos


spam_params = {
    "SP": {
        "+": jnp.array([0.95]),
        "-": jnp.array([0.95]),
        "r": jnp.array([0.95]),
        "l": jnp.array([0.95]),
        "0": jnp.array([0.95]),
        "1": jnp.array([0.95]),
    },
    "AM": {
        "X": {"prob_10": jnp.array([0.05]), "prob_01": jnp.array([0.05])},
        "Y": {"prob_10": jnp.array([0.05]), "prob_01": jnp.array([0.05])},
        "Z": {"prob_10": jnp.array([0.05]), "prob_01": jnp.array([0.05])},
    },
}

init_spam_params, unflatten_fn = ravel_pytree(spam_params)


def init_fn(rng, shape):
    return init_spam_params


class UnitarySPAMModel(nn.Module):
    hidden_sizes: list[int]
    NUM_UNITARY_PARAMS: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = UnitaryModel(
            hidden_sizes=self.hidden_sizes, NUM_UNITARY_PARAMS=self.NUM_UNITARY_PARAMS
        )(x)

        spam_params = self.param(
            "spam_params",
            lambda rng, shape: init_fn(rng, shape),
            init_spam_params.shape,
        )

        return {"model_params": x, "spam_params": unflatten_fn(spam_params)}


@deprecated.deprecated(reason="use make_loss_fn instead")
def loss_fn(
    params: VariableDict,
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    model: nn.Module,
    predictive_fn: typing.Callable,
    loss_metric: LossMetric,
    calculate_metric_fn: typing.Callable = calculate_metric,
    **model_kwargs,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """This function implement a unified interface for nn.Module.

    Args:
        params (VariableDict): Model parameters to be optimized
        control_parameters (jnp.ndarray): Control parameters parametrized Hamiltonian
        unitaries (jnp.ndarray): The Ideal unitary operators corresponding to the control parameters
        expectation_values (jnp.ndarray): Experimental expectation values to calculate the loss value
        model (nn.Module): Flax linen Blackbox part of the graybox model.
        predictive_fn (typing.Callable): Function for calculating expectation value from the model
        loss_metric (LossMetric): The choice of loss value to be minimized.
        calculate_metric_fn (typing.Callable): Function for metrics calculation from prediction and experimental value. Defaults to calculate_metric

    Returns:
        tuple[jnp.ndarray, dict[str, jnp.ndarray]]: The loss value and other metrics.
    """
    # Calculate the metrics
    predicted_expectation_value = predictive_fn(
        model=model,
        model_params=params,
        control_parameters=control_parameters,
        unitaries=unitaries,
        **model_kwargs,
    )

    metrics = calculate_metric_fn(
        unitaries, expectation_values, predicted_expectation_value
    )

    # Take mean of all the metrics
    metrics = jax.tree.map(jnp.mean, metrics)

    # ! Grab the metric in the `metrics`
    loss = metrics[loss_metric]

    return (loss, metrics)


def wo_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    # The model to be used for prediction
    model: nn.Module,
    model_params: VariableDict,
    # model keyword arguments
    **model_kwargs,
):
    """To Calculate the metrics of the model
    1. MSE Loss between the predicted expectation values and the experimental expectation values
    2. Average Gate Fidelity between the Pauli matrices to the Wo_model matrices
    3. AGF Loss between the prediction from model and the experimental expectation values

    Args:
        model (sq.model.nn.Module): The model to be used for prediction
        model_params (sq.model.VariableDict): The model parameters
        control_parameters (jnp.ndarray): The pulse parameters
        unitaries (jnp.ndarray): Ideal unitaries
        expectation_values (jnp.ndarray): Experimental expectation values
        model_kwargs (dict): Model keyword arguments
    """

    # Calculate Wo_params
    Wo = model.apply(model_params, control_parameters, **model_kwargs)

    return observable_to_expvals(Wo, unitaries)


def noisy_unitary_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    # The model to be used for prediction
    model: UnitaryModel,
    model_params: VariableDict,
    # model keyword arguments
    **model_kwargs,
):
    """Caculate for unitary-based Blackbox model

    Args:
        model (sq.model.nn.Module): The model to be used for prediction
        model_params (sq.model.VariableDict): The model parameters
        control_parameters (jnp.ndarray): The pulse parameters
        unitaries (jnp.ndarray): Ideal unitaries
        expectation_values (jnp.ndarray): Experimental expectation values
        model_kwargs (dict): Model keyword arguments

    Returns:
        typing.Any: _description_
    """

    # Predict Unitary parameters
    unitary_params = model.apply(model_params, control_parameters, **model_kwargs)

    return unitary_to_expvals(unitary_params, unitaries)


def toggling_unitary_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    # The model to be used for prediction
    model: UnitaryModel,
    model_params: VariableDict,
    # model keyword arguments
    ignore_spam: bool = False,
    **model_kwargs,
):
    """Calcuate for unitary-based Blackbox model

    Args:
        model (sq.model.nn.Module): The model to be used for prediction
        model_params (sq.model.VariableDict): The model parameters
        control_parameters (jnp.ndarray): The pulse parameters
        unitaries (jnp.ndarray): Ideal unitaries
        expectation_values (jnp.ndarray): Experimental expectation values
        model_kwargs (dict): Model keyword arguments

    Returns:
        typing.Any: _description_
    """

    # Predict Unitary parameters
    unitary_params = model.apply(model_params, control_parameters, **model_kwargs)

    if not ignore_spam:
        return toggling_unitary_with_spam_to_expvals(
            output={
                "model_params": unitary_params,
                "spam_params": model_params["spam"],
            },
            unitaries=unitaries,
        )
    else:
        return toggling_unitary_to_expvals(
            unitary_params,  # type: ignore
            unitaries=unitaries,
        )


@deprecated.deprecated
def make_loss_fn_old(
    predictive_fn: typing.Callable,
    model: nn.Module,
    calculate_metric_fn: typing.Callable = calculate_metric,
    loss_metric: LossMetric = LossMetric.MSEE,
):
    """_summary_

    Args:
        predictive_fn (typing.Callable): Function for calculating expectation value from the model
        model (nn.Module): Flax linen Blackbox part of the graybox model.
        loss_metric (LossMetric): The choice of loss value to be minimized. Defaults to LossMetric.MSEE.
        calculate_metric_fn (typing.Callable): Function for metrics calculation from prediction and experimental value. Defaults to calculate_metric.
    """

    def loss_fn(
        params: VariableDict,
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        **model_kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """This function implement a unified interface for nn.Module.

        Args:
            params (VariableDict): Model parameters to be optimized
            control_parameters (jnp.ndarray): Control parameters parametrized Hamiltonian
            unitaries (jnp.ndarray): The Ideal unitary operators corresponding to the control parameters
            expectation_values (jnp.ndarray): Experimental expectation values to calculate the loss value

        Returns:
            tuple[jnp.ndarray, dict[str, jnp.ndarray]]: The loss value and other metrics.
        """
        # Calculate the metrics
        predicted_expectation_value = predictive_fn(
            model=model,
            model_params=params,
            control_parameters=control_parameters,
            unitaries=unitaries,
            **model_kwargs,
        )

        metrics = calculate_metric_fn(
            unitaries, expectation_values, predicted_expectation_value
        )

        # Take mean of all the metrics
        metrics = jax.tree.map(jnp.mean, metrics)

        # ! Grab the metric in the `metrics`
        loss = metrics[loss_metric]

        return (loss, metrics)

    return loss_fn


def make_loss_fn(
    adapter_fn: typing.Callable,
    model: nn.Module,
    calculate_metric_fn: typing.Callable = calculate_metric,
    loss_metric: LossMetric = LossMetric.MSEE,
):
    """_summary_

    Args:
        predictive_fn (typing.Callable): Function for calculating expectation value from the model
        model (nn.Module): Flax linen Blackbox part of the graybox model.
        loss_metric (LossMetric): The choice of loss value to be minimized. Defaults to LossMetric.MSEE.
        calculate_metric_fn (typing.Callable): Function for metrics calculation from prediction and experimental value. Defaults to calculate_metric.
    """

    def loss_fn(
        params: VariableDict,
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        **model_kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """This function implement a unified interface for nn.Module.

        Args:
            params (VariableDict): Model parameters to be optimized
            control_parameters (jnp.ndarray): Control parameters parametrized Hamiltonian
            unitaries (jnp.ndarray): The Ideal unitary operators corresponding to the control parameters
            expectation_values (jnp.ndarray): Experimental expectation values to calculate the loss value

        Returns:
            tuple[jnp.ndarray, dict[str, jnp.ndarray]]: The loss value and other metrics.
        """
        output = model.apply(params, control_parameters, **model_kwargs)
        predicted_expectation_value = adapter_fn(output, unitaries=unitaries)

        # Calculate the metrics
        metrics = calculate_metric_fn(
            unitaries, expectation_values, predicted_expectation_value
        )

        # Take mean of all the metrics
        metrics = jax.tree.map(jnp.mean, metrics)

        # ! Grab the metric in the `metrics`
        loss = metrics[loss_metric]

        return (loss, metrics)

    return loss_fn


adapter_fn_type = typing.Callable[[typing.Any, jnp.ndarray], jnp.ndarray]


def make_predictive_fn(
    adapter_fn: adapter_fn_type, model: nn.Module, model_params: typing.Any
):
    def predictive_fn(
        control_parameters: jnp.ndarray, unitaries: jnp.ndarray
    ) -> jnp.ndarray:
        output = model.apply(model_params, control_parameters)

        return adapter_fn(output, unitaries)

    return predictive_fn


def create_step(
    optimizer: optax.GradientTransformation,
    loss_fn: (
        typing.Callable[..., jnp.ndarray]
        | typing.Callable[..., typing.Tuple[jnp.ndarray, typing.Any]]
    ),
    has_aux: bool = False,
):
    """The create_step function creates a training step function and a test step function.

    loss_fn should have the following signature:
    ```py
    def loss_fn(params: jaxtyping.PyTree, *args) -> jnp.ndarray:
        ...
        return loss_value
    ```
    where `params` is the parameters to be optimized, and `args` are the inputs for the loss function.

    Args:
        optimizer (optax.GradientTransformation): `optax` optimizer.
        loss_fn (typing.Callable[[jaxtyping.PyTree, ...], jnp.ndarray]): Loss function, which takes in the model parameters, inputs, and targets, and returns the loss value.
        has_aux (bool, optional): Whether the loss function return aux data or not. Defaults to False.

    Returns:
        _typing.Any_: train_step, test_step
    """

    # * Generalized training step
    @jax.jit
    def train_step(
        params: jaxtyping.PyTree,
        optimizer_state: optax.OptState,
        *args,
        **kwargs,
    ):
        loss_value, grads = jax.value_and_grad(loss_fn, has_aux=has_aux)(
            params, *args, **kwargs
        )
        updates, opt_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    @jax.jit
    def test_step(
        params: jaxtyping.PyTree,
        *args,
        **kwargs,
    ):
        return loss_fn(params, *args, **kwargs)

    return train_step, test_step


def train_model(
    # Random key
    key: jnp.ndarray,
    # Data
    train_data: DataBundled,
    val_data: DataBundled,
    test_data: DataBundled,
    # Model to be used for training
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    # Loss function to be used
    loss_fn: typing.Callable,
    # Callbacks to be used
    callbacks: list[typing.Callable] = [],
    # Number of epochs
    NUM_EPOCH: int = 1_000,
    # Optional state
    model_params: VariableDict | None = None,
    opt_state: optax.OptState | None = None,
):
    """Train the BlackBox model

    Examples:
        >>> # The number of epochs break down
        ... NUM_EPOCH = 150
        ... # Total number of iterations as 90% of data is used for training
        ... # 10% of the data is used for testing
        ... total_iterations = 9 * NUM_EPOCH
        ... # The step for optimizer if set to 8 * NUM_EPOCH (should be less than total_iterations)
        ... step_for_optimizer = 8 * NUM_EPOCH
        ... optimizer = get_default_optimizer(step_for_optimizer)
        ... # The warmup steps for the optimizer
        ... warmup_steps = 0.1 * step_for_optimizer
        ... # The cool down steps for the optimizer
        ... cool_down_steps = total_iterations - step_for_optimizer
        ... total_iterations, step_for_optimizer, warmup_steps, cool_down_steps

    Args:
        key (jnp.ndarray): Random key
        model (nn.Module): The model to be used for training
        optimizer (optax.GradientTransformation): The optimizer to be used for training
        loss_fn (typing.Callable): The loss function to be used for training
        callbacks (list[typing.Callable], optional): list of callback functions. Defaults to [].
        NUM_EPOCH (int, optional): The number of epochs. Defaults to 1_000.

    Returns:
        tuple: The model parameters, optimizer state, and the histories
    """

    key, loader_key, init_key = jax.random.split(key, 3)

    train_p, train_u, train_ex = (
        train_data.control_params,
        train_data.unitaries,
        train_data.observables,
    )
    val_p, val_u, val_ex = (
        val_data.control_params,
        val_data.unitaries,
        val_data.observables,
    )
    test_p, test_u, test_ex = (
        test_data.control_params,
        test_data.unitaries,
        test_data.observables,
    )

    BATCH_SIZE = val_p.shape[0]

    if model_params is None:
        # Initialize the model parameters if it is None
        model_params = model.init(init_key, train_p[0])

    if opt_state is None:
        # Initalize the optimizer state if it is None
        opt_state = optimizer.init(model_params)

    # histories: list[dict[str, typing.Any]] = []
    histories: list[HistoryEntryV3] = []

    train_step, eval_step = create_step(
        optimizer=optimizer, loss_fn=loss_fn, has_aux=True
    )

    for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (train_p, train_u, train_ex),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
        model_params, opt_state, (loss, aux) = train_step(
            model_params, opt_state, batch_p, batch_u, batch_ex
        )

        histories.append(HistoryEntryV3(step=step, loss=loss, loop="train", aux=aux))

        if is_last_batch:
            # Validation
            (val_loss, aux) = eval_step(model_params, val_p, val_u, val_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=val_loss, loop="val", aux=aux)
            )

            # Testing
            (test_loss, aux) = eval_step(model_params, test_p, test_u, test_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=test_loss, loop="test", aux=aux)
            )

            for callback in callbacks:
                callback(model_params, opt_state, histories)

    return model_params, opt_state, histories
