import jax
import jax.numpy as jnp
from flax import nnx
import typing
from ..model import (
    Wo_2_level_v3,
    get_predict_expectation_value,
    LossMetric,
    calculate_metric,
    unitary,
    get_spam,
)
from ..optimize import DataBundled
from ..constant import default_expectation_values_order, X, Y, Z


class Blackbox(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()

    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any:
        raise NotImplementedError()


class WoModel(Blackbox):
    def __init__(
        self, shared_layers: list[int], pauli_layers: list[int], *, rngs: nnx.Rngs
    ):
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

            observables[pauli] = Wo_2_level_v3(unitary_param, diagonal_param)

        return observables


def wo_predictive_fn(model: WoModel, data: DataBundled):
    output = model(data.control_params)
    return get_predict_expectation_value(
        output, data.unitaries, default_expectation_values_order
    )


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


class UnitaryModel(Blackbox):
    def __init__(self, hidden_sizes: list[int], *, rngs: nnx.Rngs) -> None:
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
    def __init__(self, unitary_model: UnitaryModel, *, rngs: nnx.Rngs) -> None:
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


def noisy_unitary_predictive_fn(model: UnitaryModel, data: DataBundled):
    unitary_params = model(data.control_params)
    U = unitary(unitary_params)

    predicted_expvals = get_predict_expectation_value(
        {"X": X, "Y": Y, "Z": Z},
        U,
        default_expectation_values_order,
    )

    return predicted_expvals


def toggling_unitary_predictive_fn(model: UnitaryModel, data: DataBundled):
    unitary_params = model(data.control_params)
    UJ: jnp.ndarray = unitary(unitary_params)  # type: ignore
    UJ_dagger = jnp.swapaxes(UJ, -2, -1).conj()

    expectation_value_order, observables = (
        default_expectation_values_order,
        {"X": X, "Y": Y, "Z": Z},
    )

    X_ = UJ_dagger @ observables["X"] @ UJ
    Y_ = UJ_dagger @ observables["Y"] @ UJ
    Z_ = UJ_dagger @ observables["Z"] @ UJ

    predicted_expvals = get_predict_expectation_value(
        {"X": X_, "Y": Y_, "Z": Z_},
        data.unitaries,
        expectation_value_order,
    )

    return predicted_expvals


def toggling_unitary_with_spam_predictive_fn(
    model: UnitarySPAMModel, data: DataBundled
):
    unitary_params = model(data.control_params)
    UJ: jnp.ndarray = unitary(unitary_params)  # type: ignore
    UJ_dagger = jnp.swapaxes(UJ, -2, -1).conj()

    expectation_value_order, observables = get_spam(model.spam_params)

    X_ = UJ_dagger @ observables["X"] @ UJ
    Y_ = UJ_dagger @ observables["Y"] @ UJ
    Z_ = UJ_dagger @ observables["Z"] @ UJ

    predicted_expvals = get_predict_expectation_value(
        {"X": X_, "Y": Y_, "Z": Z_},
        data.unitaries,
        expectation_value_order,
    )

    return predicted_expvals


def make_loss_fn(
    predictive_fn,
    calculate_metric_fn=calculate_metric,
    loss_metric: LossMetric = LossMetric.MSEE,
):
    def loss_fn(model: Blackbox, data: DataBundled):
        expval = predictive_fn(model, data)

        metrics = calculate_metric_fn(data.unitaries, data.observables, expval)
        # Take mean of all the metrics
        metrics = jax.tree.map(jnp.mean, metrics)
        loss = metrics[loss_metric]

        return loss, metrics

    return loss_fn


def create_step(
    loss_fn: typing.Callable[[Blackbox, DataBundled], tuple[jnp.ndarray, typing.Any]],
):
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
    abstract_model = nnx.eval_shape(lambda: Model(**config, rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    nnx.replace_by_pure_dict(abstract_state, model_params)

    return nnx.merge(graphdef, abstract_state)
