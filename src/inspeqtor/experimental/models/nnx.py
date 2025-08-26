import jax
import jax.numpy as jnp
from flax import nnx
import typing
import optax

from inspeqtor.experimental.decorator import warn_not_tested_function
from ..model import (
    hermitian,
    LossMetric,
    calculate_metric,
    observable_to_expvals,
    unitary_to_expvals,
    toggling_unitary_to_expvals,
    toggling_unitary_with_spam_to_expvals,
)
from ..optimize import DataBundled, HistoryEntryV3
from ..utils import dataloader


class Blackbox(nnx.Module):
    """The abstract class for interfacing the Blackbox model of the Graybox"""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()

    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any:
        raise NotImplementedError()


class WoModel(Blackbox):
    """$\\hat{W}_{O}$ based blackbox model."""

    def __init__(
        self, shared_layers: list[int], pauli_layers: list[int], *, rngs: nnx.Rngs
    ):
        """
        Args:
            shared_layers (list[int]): Each integer in the list is a size of the width of each hidden layer in the shared layers.
            pauli_layers (list[int]): Each integer in the list is a size of the width of each hidden layer in the Pauli layers.
            rngs (nnx.Rngs): Random number generator of `nnx`.
        """
        self.shared_layers = [
            nnx.Linear(in_features=in_features, out_features=out_features, rngs=rngs)
            for in_features, out_features in zip(shared_layers[:-1], shared_layers[1:])
        ]
        self.pauli_layers = {}
        self.unitary_layers = {}
        self.diagonal_layers = {}
        for pauli in ["X", "Y", "Z"]:
            layers = [
                nnx.Linear(
                    in_features=in_features, out_features=out_features, rngs=rngs
                )
                for in_features, out_features in zip(
                    pauli_layers[:-1], pauli_layers[1:]
                )
            ]
            self.pauli_layers[pauli] = layers

            self.unitary_layers[pauli] = nnx.Linear(
                in_features=pauli_layers[-1], out_features=3, rngs=rngs
            )
            self.diagonal_layers[pauli] = nnx.Linear(
                in_features=pauli_layers[-1], out_features=2, rngs=rngs
            )

    def __call__(self, x: jnp.ndarray):
        for layer in self.shared_layers:
            x = nnx.relu(layer(x))

        observables: dict[str, jnp.ndarray] = dict()
        for pauli, pauli_layer in self.pauli_layers.items():
            _x = jnp.copy(x)
            for layer in pauli_layer:
                _x = nnx.relu(layer(_x))

            unitary_param = self.unitary_layers[pauli](_x)
            diagonal_param = self.diagonal_layers[pauli](_x)

            unitary_param = 2 * jnp.pi * nnx.hard_sigmoid(unitary_param)
            diagonal_param = (2 * nnx.hard_sigmoid(diagonal_param)) - 1

            observables[pauli] = hermitian(unitary_param, diagonal_param)

        return observables


def wo_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    model: WoModel,
) -> jnp.ndarray:
    """Adapter function for $\\hat{W}_{O}$ based model to be used with `make_loss_fn`.

    Args:
        model (WoModel): $\\hat{W}_{O}$ based model
        data (DataBundled): A bundled of data for the predictive model training.

    Returns:
        jnp.ndarray: Predicted expectation values.
    """
    output = model(control_parameters)
    return observable_to_expvals(output, unitaries)


class UnitaryModel(Blackbox):
    """Unitary-based model, predicting parameters parametrized unitary operator in range $[0, 2\\pi]$."""

    def __init__(self, hidden_sizes: list[int], *, rngs: nnx.Rngs) -> None:
        """

        Args:
            hidden_sizes (list[int]): Each integer in the list is a size of the width of each hidden layer in the shared layers
            rngs (nnx.Rngs): Random number generator of `nnx`.
        """
        self.hidden_sizes = hidden_sizes
        self.NUM_UNITARY_PARAMS = 4

        # Initialize the dense layers for each hidden size
        self.hidden_layers = [
            nnx.Linear(in_features=hidden_size, out_features=hidden_size, rngs=rngs)
            for hidden_size in self.hidden_sizes
        ]

        # Initialize the final layer for unitary parameters
        self.final_layer = nnx.Linear(
            in_features=self.hidden_sizes[-1],
            out_features=self.NUM_UNITARY_PARAMS,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply the hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        # Apply the final layer and transform the output
        x = self.final_layer(x)
        x = 2 * jnp.pi * nnx.hard_sigmoid(x)

        return x


class UnitarySPAMModel(Blackbox):
    """Composite class of unitary-based model and the SPAM model."""

    def __init__(self, unitary_model: UnitaryModel, *, rngs: nnx.Rngs) -> None:
        """

        Args:
            unitary_model (UnitaryModel): Unitary-based model that have already initialized.
            rngs (nnx.Rngs): Random number generator of `nnx`.
        """
        self.unitary_model = unitary_model
        self.spam_params = {
            "SP": {
                "+": nnx.Param(jnp.array([0.9])),
                "-": nnx.Param(jnp.array([0.9])),
                "r": nnx.Param(jnp.array([0.9])),
                "l": nnx.Param(jnp.array([0.9])),
                "0": nnx.Param(jnp.array([0.9])),
                "1": nnx.Param(jnp.array([0.9])),
            },
            "AM": {
                "X": {
                    "prob_10": nnx.Param(jnp.array([0.1])),
                    "prob_01": nnx.Param(jnp.array([0.1])),
                },
                "Y": {
                    "prob_10": nnx.Param(jnp.array([0.1])),
                    "prob_01": nnx.Param(jnp.array([0.1])),
                },
                "Z": {
                    "prob_10": nnx.Param(jnp.array([0.1])),
                    "prob_01": nnx.Param(jnp.array([0.1])),
                },
            },
        }

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.unitary_model(x)


def noisy_unitary_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    model: UnitaryModel,
) -> jnp.ndarray:
    """Adapter function for unitary-based model to be used with `make_loss_fn`

    Args:
        model (UnitaryModel): Unitary-based model.
        data (DataBundled): A bundled of data for the predictive model training.

    Returns:
        jnp.ndarray: Predicted expectation values.
    """
    unitary_params = model(control_parameters)

    return unitary_to_expvals(unitary_params, unitaries)


def toggling_unitary_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    model: UnitaryModel,
) -> jnp.ndarray:
    """Adapter function for rotating toggling frame unitary based model to be used with `make_loss_fn`

    Args:
        model (UnitaryModel): Unitary-based model.
        data (DataBundled): A bundled of data for the predictive model training.

    Returns:
        jnp.ndarray: Predicted expectation values.
    """
    unitary_params = model(control_parameters)

    return toggling_unitary_to_expvals(unitary_params, unitaries)


def toggling_unitary_with_spam_predictive_fn(
    # Input data to the model
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    model: UnitarySPAMModel,
) -> jnp.ndarray:
    """Adapter function for a composite rotating toggling
    frame unitary based model to be used with `make_loss_fn`

    Args:
        model (UnitaryModel): Unitary-based SPAM model.
        data (DataBundled): A bundled of data for the predictive model training.

    Returns:
        jnp.ndarray: Predicted expectation values.
    """
    unitary_params = model(control_parameters)

    return toggling_unitary_with_spam_to_expvals(
        {
            "model_params": unitary_params,
            "spam_params": model.spam_params,
        },
        unitaries,
    )


def make_loss_fn(
    predictive_fn,
    calculate_metric_fn=calculate_metric,
    loss_metric: LossMetric = LossMetric.MSEE,
):
    """A function for preparing loss function to be used for model training.

    Args:
        predictive_fn (typing.Any): Adaptor function specifically for each model.
        calculate_metric_fn (typing.Any, optional): Function for calculating metrics. Defaults to calculate_metric.
        loss_metric (LossMetric, optional): The chosen loss function to be optimized. Defaults to LossMetric.MSEE.
    """

    def loss_fn(model: Blackbox, data: DataBundled):
        expval = predictive_fn(data.control_params, data.unitaries, model)

        metrics = calculate_metric_fn(data.unitaries, data.observables, expval)
        # Take mean of all the metrics
        metrics = jax.tree.map(jnp.mean, metrics)
        loss = metrics[loss_metric]

        return loss, metrics

    return loss_fn


def create_step(
    loss_fn: typing.Callable[[Blackbox, DataBundled], tuple[jnp.ndarray, typing.Any]],
):
    """A function to create the traning and evaluating step for model.
    The train step will update the model parameters and optimizer parameters inplace.

    Args:
        loss_fn (typing.Callable[[Blackbox, DataBundled], tuple[jnp.ndarray, typing.Any]]): Loss function returned from `make_loss_fn`

    Returns:
        typing.Any: The tuple of training and eval step functions.
    """

    @nnx.jit
    def train_step(
        model: Blackbox,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        data: DataBundled,
    ):
        """Train for a single step."""
        model.train()  # Switch to train mode
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model, data)
        metrics.update(loss=loss)  # In-place updates.
        optimizer.update(grads)  # In-place updates.

        return loss, aux

    @nnx.jit
    def eval_step(model: Blackbox, metrics: nnx.MultiMetric, data):
        model.eval()
        loss, aux = loss_fn(model, data)
        metrics.update(loss=loss)  # In-place updates.

        return loss, aux

    return train_step, eval_step


T = typing.TypeVar("T", bound=Blackbox)


def reconstruct_model(model_params, config, Model: type[T]) -> T:
    """Reconstruct the model from the model parameters, config, and model initializer.

    Examples:
        >>> _, state = nnx.split(blackbox)
        >>> model_params = nnx.to_pure_dict(state)
        >>> config = {
        ...    "shared_layers": [8],
        ...    "pauli_layers": [8]
        ... }
        >>> model_data = sq.model.ModelData(params=model_params, config=config)
        # save and load to and from disk!
        >>> blackbox = sq.models.nnx.reconstruct_model(model_data.params, model_data.config, sq.models.nnx.WoModel)

    Args:
        model_params (typing.Any): The pytree containing model parameters.
        config (typing.Any): The model configuration for model initialization.
        Model (type[T]): The model initializer.

    Returns:
        T: _description_
    """
    abstract_model = nnx.eval_shape(lambda: Model(**config, rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, model_params)

    return nnx.merge(graphdef, abstract_state)


@warn_not_tested_function
def train_model(
    # Random key
    key: jnp.ndarray,
    # Data
    train_data: DataBundled,
    val_data: DataBundled,
    test_data: DataBundled,
    # Model to be used for training
    model: Blackbox,
    optimizer: optax.GradientTransformation,
    # Loss function to be used
    loss_fn: typing.Callable,
    # Callbacks to be used
    callbacks: list[typing.Callable] = [],
    # Number of epochs
    NUM_EPOCH: int = 1_000,
    _optimizer: nnx.Optimizer | None = None,
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

    key, loader_key = jax.random.split(key)

    BATCH_SIZE = val_data.control_params.shape[0]

    histories: list[HistoryEntryV3] = []

    if _optimizer is None:
        _optimizer = nnx.Optimizer(
            model,
            optimizer,
            wrt=nnx.Param,
        )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    train_step, eval_step = create_step(loss_fn=loss_fn)

    for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (
            train_data.control_params,
            train_data.unitaries,
            train_data.observables,
        ),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
        train_step(
            model,
            _optimizer,
            metrics,
            DataBundled(
                control_params=batch_p, unitaries=batch_u, observables=batch_ex
            ),
        )

        histories.append(
            HistoryEntryV3(
                step=step, loss=metrics.compute()["loss"], loop="train", aux={}
            )
        )
        metrics.reset()  # Reset the metrics for the train set.

        if is_last_batch:
            # Validation
            eval_step(model, metrics, val_data)
            histories.append(
                HistoryEntryV3(
                    step=step, loss=metrics.compute()["loss"], loop="val", aux={}
                )
            )
            metrics.reset()  # Reset the metrics for the val set.
            # Testing
            eval_step(model, metrics, test_data)
            histories.append(
                HistoryEntryV3(
                    step=step, loss=metrics.compute()["loss"], loop="test", aux={}
                )
            )
            metrics.reset()  # Reset the metrics for the test set.

            for callback in callbacks:
                callback(model, _optimizer, histories)

    return model, _optimizer, histories
