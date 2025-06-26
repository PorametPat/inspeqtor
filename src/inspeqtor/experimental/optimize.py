import typing
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.typing import VariableDict
from dataclasses import dataclass
import flax.traverse_util as traverse_util
from functools import partial
import optax  # type: ignore
from alive_progress import alive_it  # type: ignore

import tempfile
from enum import StrEnum
import chex

from .control import ControlSequence
from .utils import dataloader, create_step
from .model import loss_fn, LossMetric, save_model, load_model
from ray.tune.search.sample import Domain

# from collections import namedtuple
# DataBundled = namedtuple(
#     "DataBundled", ["control_params", "unitaries", "observables", "aux"]
# )

@dataclass
class DataBundled:
    control_params: jnp.ndarray
    unitaries: jnp.ndarray
    observables: jnp.ndarray
    aux: jnp.ndarray | None = None


def get_default_optimizer(n_iterations: int) -> optax.GradientTransformation:
    """Generate present optimizer from number of training iteration.

    Args:
        n_iterations (int): Training iteration

    Returns:
        optax.GradientTransformation: Optax optimizer.
    """
    return optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=int(0.1 * n_iterations),
            decay_steps=n_iterations,
            end_value=1e-6,
        )
    )


def minimize(
    params: chex.ArrayTree,
    func: typing.Callable[[jnp.ndarray], tuple[jnp.ndarray, typing.Any]],
    optimizer: optax.GradientTransformation,
    lower: chex.ArrayTree | None = None,
    upper: chex.ArrayTree | None = None,
    maxiter: int = 1000,
) -> tuple[chex.ArrayTree, list[typing.Any]]:
    """Optimize the loss function with bounded parameters.

    Args:
        params (chex.ArrayTree): Intiial parameters to be optimized
        lower (chex.ArrayTree): Lower bound of the parameters
        upper (chex.ArrayTree): Upper bound of the parameters
        func (typing.Callable[[jnp.ndarray], tuple[jnp.ndarray, typing.Any]]): Loss function
        optimizer (optax.GradientTransformation): Instance of optax optimizer
        maxiter (int, optional): Number of optimization step. Defaults to 1000.

    Returns:
        tuple[chex.ArrayTree, list[typing.Any]]: Tuple of parameters and optimization history
    """
    opt_state = optimizer.init(params)
    history = []

    for _ in alive_it(range(maxiter), force_tty=True):
        grads, aux = jax.grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if lower is not None and upper is not None:
            # Apply projection
            params = optax.projections.projection_box(params, lower, upper)

        # Log the history
        aux["params"] = params
        history.append(aux)

    return params, history


@dataclass
class HistoryEntryV3:
    step: int
    loss: float | jnp.ndarray
    loop: str
    aux: dict[str, jnp.ndarray]


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
):
    """Train the BlackBox model

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

    # Initialize the model parameters
    model_params = model.init(init_key, train_p[0])
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


def transform_key(data):
    return {
        # Concanate the key by '/'
        "/".join(key): value
        for key, value in data.items()
    }


def clean_history_entries(
    histories: list[HistoryEntryV3],
):
    clean_histories = [
        {
            "step": history.step,
            "loss": history.loss,
            "loop": history.loop,
            **history.aux,
        }
        for history in histories
    ]
    # Move from device to host, i.e. from jax.Array to numpy.ndarray
    clean_histories = jax.tree.map(
        lambda x: x.item() if isinstance(x, jnp.ndarray) else x, clean_histories
    )
    # Flatten the nested dictionaries
    clean_histories = list(map(traverse_util.flatten_dict, clean_histories))
    # Transform the keys of the dictionary
    clean_histories = list(map(transform_key, clean_histories))
    return clean_histories


def default_trainable_v4(
    control_sequence: ControlSequence,
    metric: LossMetric,
    experiment_identifier: str,
    hamiltonian: typing.Callable | str,
    construct_model_fn: typing.Callable[
        [dict[str, int]], tuple[nn.Module, dict[str, typing.Any]]
    ],
    calculate_metrics_fn: typing.Callable,
    NUM_EPOCH: int = 1000,
    CHECKPOINT_EVERY: int = 100,
):
    """Create trainable function for `ray.tune` for hyperparameter tuning

    Args:
        control_sequence (PulseSequence): Pulse sequence of dataset
        metric (LossMetric): Metric to be minimized for.
        experiment_identifier (str): The experiment identifier
        hamiltonian (typing.Callable | str): Ideal Hamiltonian function or name.
        model_choice (type[nn.Module], optional): Choice of the Blackbox model. Defaults to BasicBlackBoxV2.
        NUM_EPOCH (int, optional): Number of training epoch. Defaults to 1000.
        CHECKPOINT_EVERY (int, optional): Checkpointing every given number. Defaults to 100.

    Returns:
        typing.Callable: Trainable function that recieve hyperparameter configutation, dataset and random key.
    """
    from ray import train

    def trainable(
        config: dict[str, int],
        train_data: DataBundled,
        val_data: DataBundled,
        test_data: DataBundled,
        train_key: jnp.ndarray,
    ):
        optimizer = get_default_optimizer(8 * NUM_EPOCH)

        model, model_config = construct_model_fn(config)

        partial_loss_fn = partial(
            loss_fn,
            model=model,
            loss_metric=metric,
            calculate_metrics_fn=calculate_metrics_fn,
        )

        def prepare_report(history: list[HistoryEntryV3]):
            metric_types = [LossMetric.MSEE, LossMetric.AEF, LossMetric.WAEE]
            metrics = {}
            for entry in history:
                for metric_type in metric_types:
                    metrics[f"{entry.loop}/{metric_type}"] = entry.aux[
                        metric_type
                    ].item()

            return metrics

        def callback(
            model_params: VariableDict,
            opt_state: optax.OptState,
            history: list[HistoryEntryV3],
        ) -> None:
            # Get the lasted 3 entries
            last_entries = history[-3:]

            loops = ["train", "val", "test"]
            # assert that the last 3 entries are from train, val, and test
            assert all(entry.loop in loops for entry in last_entries)

            # Prepare the report
            metrics = prepare_report(history)

            # Check if last_entry.step is divisible by 100
            if (last_entries[-1].step + 1) % CHECKPOINT_EVERY == 0:
                # Checkpoint the model

                # Clean the history entries
                clean_histories = clean_history_entries(history)

                with tempfile.TemporaryDirectory() as tmpdir:
                    _ = save_model(
                        path=tmpdir,
                        experiment_identifier=experiment_identifier,
                        control_sequence=control_sequence,
                        hamiltonian=hamiltonian,
                        model_config=model_config,
                        model_params=model_params,
                        history=clean_histories,
                        with_auto_datetime=False,
                    )

                    # Report the loss and val_loss to tune
                    train.report(
                        metrics=metrics,
                        checkpoint=train.Checkpoint.from_directory(tmpdir),
                    )
            else:
                # Report the loss and val_loss to tune
                train.report(
                    metrics=metrics,
                )

            return None

        _, _, history = train_model(
            key=train_key,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            model=model,
            optimizer=optimizer,
            loss_fn=partial_loss_fn,
            NUM_EPOCH=NUM_EPOCH,
            callbacks=[callback],
        )

        # Prepare the report
        metrics = prepare_report(history[-3:])

        return metrics

    return trainable


def sample_from_search_space(search_space: typing.Mapping[str, Domain]):
    return {key: value.sample() for key, value in search_space.items()}


class SearchAlgo(StrEnum):
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"


def hypertuner(
    trainable: typing.Callable,
    train_data: DataBundled,
    test_data: DataBundled,
    val_data: DataBundled,
    train_key: jnp.ndarray,
    metric: LossMetric,
    search_space: typing.Mapping[str, Domain],
    num_samples: int = 100,
    search_algo: SearchAlgo = SearchAlgo.HYPEROPT,
):
    """Perform hyperparameter tuning

    Args:
        trainable (typing.Callable): Trainable function
        train_pulse_parameters (jnp.ndarray): Training pulse parameters
        train_unitaries (jnp.ndarray): Training ideal unitary matrix
        train_expectation_values (jnp.ndarray): Training experiment expectation value
        test_pulse_parameters (jnp.ndarray): Testing pulse parameters
        test_unitaries (jnp.ndarray): Testing ideal unitary matrix
        test_expectation_values (jnp.ndarray): Testing experiment expectation value
        val_pulse_parameters (jnp.ndarray): Validating pulse parameters
        val_unitaries (jnp.ndarray): Validating ideal unitary matrix
        val_expectation_values (jnp.ndarray): Validating experiment expectation value
        train_key (jnp.ndarray): Random key
        metric (LossMetric): Metric to optimized for.
        num_samples (int, optional): The number of random configuration of hyperparameter. Defaults to 100.
        search_algo (SearchAlgo, optional): The search algorithm to be used for optimization. Defaults to SearchAlgo.HYPEROPT.
        search_spaces (_type_, optional): Search space of hyperparameters. Defaults to { "hidden_layer_1_1": (5, 50), "hidden_layer_1_2": (5, 50), "hidden_layer_2_1": (5, 50), "hidden_layer_2_2": (5, 50), }.
        initial_config (_type_, optional): Initial hyperparameters. Defaults to { "hidden_layer_1_1": 10, "hidden_layer_1_2": 20, "hidden_layer_2_1": 10, "hidden_layer_2_2": 20, }.

    Returns:
        _type_: Optimization result.
    """
    from ray import tune
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import Searcher

    current_best_params = [{key: value.sample() for key, value in search_space.items()}]

    # Prepend 'val/' to the metric
    prepended_metric = f"val/{metric}"

    if search_algo == SearchAlgo.HYPEROPT:
        search_algo_instance: Searcher = HyperOptSearch(
            metric=prepended_metric,
            mode="min",
            points_to_evaluate=current_best_params,
        )
    elif search_algo == SearchAlgo.OPTUNA:
        search_algo_instance = OptunaSearch(
            metric=prepended_metric,
            mode="min",
        )

    run_config = tune.RunConfig(
        name="tune_experiment",
        checkpoint_config=tune.CheckpointConfig(
            num_to_keep=10,
        ),
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            trainable,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_key=train_key,
        ),
        tune_config=tune.TuneConfig(
            search_alg=search_algo_instance,
            metric=prepended_metric,
            mode="min",
            num_samples=num_samples,
        ),
        param_space=search_space,  # type: ignore
        run_config=run_config,
    )

    results = tuner.fit()

    return results


def get_best_hypertuner_results(results, metric: LossMetric, loop: str = "val"):
    prepended_metric = f"{loop}/{metric}"

    with results.get_best_result(
        metric=prepended_metric, mode="min"
    ).checkpoint.as_directory() as checkpoint_dir:
        model_state, hist, data_config = load_model(checkpoint_dir, skip_history=False)
    return model_state, hist, data_config
