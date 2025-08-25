import deprecated
import jax
import jax.numpy as jnp
import optax
import typing
import jaxtyping
from flax import linen as nn
from flax.typing import VariableDict

from ..model import (
    LossMetric,
    get_predict_expectation_value,
    calculate_metric,
    Wo_2_level_v3,
    unitary,
    get_spam,
)

from ..optimize import DataBundled, HistoryEntryV3
from ..utils import dataloader
from ..constant import (
    default_expectation_values_order,
    X,
    Y,
    Z,
)


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
        unitary_activation_fn (typing.Any, optional): Activation function for unitary parameters. Defaults to lambdax:2*jnp.pi*nn.hard_sigmoid(x).
        diagonal_activation_fn (typing.Any, optional): Activation function for diagonal parameters. Defaults to lambdax:(2 * nn.hard_sigmoid(x))-1.

    Returns:
        type[nn.Module]: Constructor of the Blackbox model
    """

    class BlackBox(nn.Module):
        hidden_sizes_1: typing.Sequence[int] = (20, 10)
        hidden_sizes_2: typing.Sequence[int] = (20, 10)
        pauli_operators: typing.Sequence[str] = ("X", "Y", "Z")

        NUM_UNITARY_PARAMS: int = 3
        NUM_DIAGONAL_PARAMS: int = 2

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
                unitary_params = unitary_activation_fn(unitary_params)
                # For the diagonal part, we use a dense layer with 1 feature
                diag_params = nn.Dense(
                    features=self.NUM_DIAGONAL_PARAMS, name=f"D_{op}"
                )(_x)
                # Apply the activation function
                diag_params = diagonal_activation_fn(diag_params)

                Wos[op] = Wo_2_level_v3(unitary_params, diag_params)

            return Wos

    return BlackBox


class WoModel(nn.Module):
    hidden_sizes_1: typing.Sequence[int] = (20, 10)
    hidden_sizes_2: typing.Sequence[int] = (20, 10)
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

            Wos[op] = Wo_2_level_v3(unitary_params, diag_params)

        return Wos


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

    # Calculate the predicted expectation values using model
    predicted_expvals = get_predict_expectation_value(
        Wo,  # type: ignore
        unitaries,
        default_expectation_values_order,
    )

    return predicted_expvals


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

    U = unitary(unitary_params)  # type: ignore

    predicted_expvals = get_predict_expectation_value(
        {"X": X, "Y": Y, "Z": Z},
        U,
        default_expectation_values_order,
    )

    return predicted_expvals


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

    UJ: jnp.ndarray = unitary(unitary_params)  # type: ignore
    UJ_dagger = jnp.swapaxes(UJ, -2, -1).conj()

    if not ignore_spam:
        expectation_value_order, observables = get_spam(model_params["spam"])
    else:
        expectation_value_order, observables = (
            default_expectation_values_order,
            {"X": X, "Y": Y, "Z": Z},
        )

    X_ = UJ_dagger @ observables["X"] @ UJ
    Y_ = UJ_dagger @ observables["Y"] @ UJ
    Z_ = UJ_dagger @ observables["Z"] @ UJ

    predicted_expvals = get_predict_expectation_value(
        {"X": X_, "Y": Y_, "Z": Z_},
        unitaries,
        expectation_value_order,
    )

    return predicted_expvals


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
        unitary_activation_fn (typing.Any, optional): Activation function for unitary parameters. Defaults to lambdax:2*jnp.pi*nn.hard_sigmoid(x).
        diagonal_activation_fn (typing.Any, optional): Activation function for diagonal parameters. Defaults to lambdax:(2 * nn.hard_sigmoid(x))-1.

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


class WoDropoutModel(nn.Module):
    dropout_rate: float = 0.2
    hidden_sizes_1: typing.Sequence[int] = (20, 10)
    hidden_sizes_2: typing.Sequence[int] = (20, 10)
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

            Wos[op] = Wo_2_level_v3(unitary_params, diag_params)

        return Wos


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


def make_loss_fn(
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
    model_params: typing.Any = None,
    opt_state: typing.Any = None,
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
