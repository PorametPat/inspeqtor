import typing
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.typing import VariableDict
from dataclasses import dataclass
import flax.traverse_util as traverse_util
from functools import partial
import optax

# from ray import tune, train
import tempfile

# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import Searcher
from enum import StrEnum

from .pulse import PulseSequence
from .utils import random_split, dataloader, create_step
from .model import BasicBlackBoxV2, loss_fn, LossMetric, save_model, load_model


def get_default_optimizer(n_iterations):
    return optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=int(0.1 * n_iterations),
            decay_steps=n_iterations,
            end_value=1e-6,
        )
    )


@dataclass
class HistoryEntryV3:
    step: int
    loss: float | jnp.ndarray
    loop: str
    aux: dict[str, jnp.ndarray]


def train_model(
    # Random key
    key: jnp.ndarray,
    # Data to be used for training and testing
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
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

    ```py
    # The number of epochs break down
    NUM_EPOCH = 150
    # Total number of iterations as 90% of data is used for training
    # 10% of the data is used for testing
    total_iterations = 9 * NUM_EPOCH
    # The step for optimizer if set to 8 * NUM_EPOCH (should be less than total_iterations)
    step_for_optimizer = 8 * NUM_EPOCH
    optimizer = get_default_optimizer(step_for_optimizer)
    # The warmup steps for the optimizer
    warmup_steps = 0.1 * step_for_optimizer
    # The cool down steps for the optimizer
    cool_down_steps = total_iterations - step_for_optimizer

    total_iterations, step_for_optimizer, warmup_steps, cool_down_steps
    ```

    Args:
        key (jnp.ndarray): Random key
        pulse_parameters (jnp.ndarray): Pulse parameters in the shape of (n_samples, n_features)
        unitaries (jnp.ndarray): Ideal unitaries in the shape of (n_samples, 2, 2)
        expectation_values (jnp.ndarray): Experimental expectation values in the shape of (n_samples, 18)
        model (nn.Module): The model to be used for training
        optimizer (optax.GradientTransformation): The optimizer to be used for training
        loss_fn (typing.Callable): The loss function to be used for training
        callbacks (list[typing.Callable], optional): list of callback functions. Defaults to [].
        NUM_EPOCH (int, optional): The number of epochs. Defaults to 1_000.

    Returns:
        tuple: The model parameters, optimizer state, and the histories
    """

    key, split_key, loader_key, init_key = jax.random.split(key, 4)

    # Split the data into train and val
    # use 10% of the data for valing
    val_size = int(0.1 * pulse_parameters.shape[0])
    train_p, train_u, train_ex, val_p, val_u, val_ex = random_split(
        split_key,
        val_size,
        pulse_parameters,
        unitaries,
        expectation_values,
    )

    # Initialize the model parameters
    model_params = model.init(init_key, pulse_parameters[0])
    opt_state = optimizer.init(model_params)

    # histories: list[dict[str, typing.Any]] = []
    histories: list[HistoryEntryV3] = []

    train_step, val_step = create_step(
        optimizer=optimizer, loss_fn=loss_fn, has_aux=True
    )

    for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (train_p, train_u, train_ex),
        batch_size=val_size,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
        model_params, opt_state, (loss, aux) = train_step(
            model_params, opt_state, batch_p, batch_u, batch_ex
        )

        histories.append(HistoryEntryV3(step=step, loss=loss, loop="train", aux=aux))

        if is_last_batch:
            (val_loss, aux) = val_step(model_params, val_p, val_u, val_ex)

            histories.append(
                HistoryEntryV3(step=step, loss=val_loss, loop="val", aux=aux)
            )

            for callback in callbacks:
                callback(model_params, opt_state, histories)

    return model_params, opt_state, histories


def transform_key(data: dict[typing.Sequence[str], typing.Any]):
    return {
        # Concanate the key by '/'
        "/".join(key): value
        for key, value in data.items()
    }


def clean_history_entries(
    histories: list[HistoryEntryV3],
) -> list[dict[str, str | float]]:
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


def default_trainable_v3(
    pulse_sequence: PulseSequence,
    metric: LossMetric,
    experiment_identifier: str,
    hamiltonian: typing.Callable | str,
    NUM_EPOCH: int = 1000,
    CHECKPOINT_EVERY: int = 100,
):
    from ray import train

    def trainable(
        config: dict[str, int],
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        train_key: jnp.ndarray,
    ):
        # FEATURE_SIZE = config["feature_size"]
        HIDDEN_LAYER_1_1 = config["hidden_layer_1_1"]
        HIDDEN_LAYER_1_2 = config["hidden_layer_1_2"]
        HIDDEN_LAYER_2_1 = config["hidden_layer_2_1"]
        HIDDEN_LAYER_2_2 = config["hidden_layer_2_2"]

        model_config: dict[str, int | list[int]] = {
            "hidden_sizes_1": [HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2],
            "hidden_sizes_2": [HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2],
        }

        optimizer = get_default_optimizer(8 * NUM_EPOCH)

        model = BasicBlackBoxV2(
            hidden_sizes_1=[HIDDEN_LAYER_1_1, HIDDEN_LAYER_1_2],
            hidden_sizes_2=[HIDDEN_LAYER_2_1, HIDDEN_LAYER_2_2],
        )

        partial_loss_fn = partial(loss_fn, model=model, loss_metric=metric)

        def callback(
            model_params: VariableDict,
            opt_state: optax.OptState,
            history: list[HistoryEntryV3],
        ) -> None:
            last_entry = history[-1]

            assert last_entry.loop == "val"

            # Check if last_entry.step is divisible by 100
            if (last_entry.step + 1) % CHECKPOINT_EVERY == 0:
                # Checkpoint the model

                # Clean the history entries
                clean_histories = clean_history_entries(history)

                with tempfile.TemporaryDirectory() as tmpdir:
                    _ = save_model(
                        path=tmpdir,
                        experiment_identifier=experiment_identifier,
                        pulse_sequence=pulse_sequence,
                        hamiltonian=hamiltonian,
                        model_config=model_config,
                        model_params=model_params,
                        history=clean_histories,
                        with_auto_datetime=False,
                    )

                    # Report the loss and val_loss to tune
                    train.report(
                        metrics={
                            f"{last_entry.loop}/{LossMetric.MSEE}": last_entry.aux[
                                LossMetric.MSEE
                            ].item(),
                            f"{last_entry.loop}/{LossMetric.AEF}": last_entry.aux[
                                LossMetric.AEF
                            ].item(),
                            f"{last_entry.loop}/{LossMetric.WAEE}": last_entry.aux[
                                LossMetric.WAEE
                            ].item(),
                        },
                        checkpoint=train.Checkpoint.from_directory(tmpdir),
                    )
            else:
                # Report the loss and val_loss to tune
                train.report(
                    metrics={
                        f"{last_entry.loop}/{LossMetric.MSEE}": last_entry.aux[
                            LossMetric.MSEE
                        ].item(),
                        f"{last_entry.loop}/{LossMetric.AEF}": last_entry.aux[
                            LossMetric.AEF
                        ].item(),
                        f"{last_entry.loop}/{LossMetric.WAEE}": last_entry.aux[
                            LossMetric.WAEE
                        ].item(),
                    },
                )

            return None

        _, _, history = train_model(
            key=train_key,
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectation_values=expectation_values,
            model=model,
            optimizer=optimizer,
            loss_fn=partial_loss_fn,
            NUM_EPOCH=NUM_EPOCH,
            callbacks=[callback],
        )

        return {
            f"val/{LossMetric.MSEE}": history[-1].aux[LossMetric.MSEE].item(),
            f"val/{LossMetric.AEF}": history[-1].aux[LossMetric.AEF].item(),
            f"val/{LossMetric.WAEE}": history[-1].aux[LossMetric.WAEE].item(),
        }

    return trainable


class SearchAlgo(StrEnum):
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"


def hypertuner(
    trainable: typing.Callable,
    pulse_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    expectation_values: jnp.ndarray,
    train_key: jnp.ndarray,
    metric: LossMetric,
    num_samples: int = 100,
    search_algo: SearchAlgo = SearchAlgo.HYPEROPT,
):
    from ray import tune, train
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import Searcher

    search_space = {
        "hidden_layer_1_1": tune.randint(5, 50),
        "hidden_layer_1_2": tune.randint(5, 50),
        "hidden_layer_2_1": tune.randint(5, 50),
        "hidden_layer_2_2": tune.randint(5, 50),
    }

    current_best_params = [
        {
            "hidden_layer_1_1": 10,
            "hidden_layer_1_2": 20,
            "hidden_layer_2_1": 10,
            "hidden_layer_2_2": 20,
        }
    ]

    # Prepend 'test/' to the metric
    prepended_metric = f"val/{metric}"

    if search_algo == SearchAlgo.HYPEROPT:
        search_algo_instance: Searcher = HyperOptSearch(
            metric=prepended_metric, mode="min", points_to_evaluate=current_best_params
        )
    elif search_algo == SearchAlgo.OPTUNA:
        search_algo_instance = OptunaSearch(metric=prepended_metric, mode="min")

    run_config = train.RunConfig(
        name="tune_experiment",
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=10,
        ),
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            trainable,
            pulse_parameters=pulse_parameters,
            unitaries=unitaries,
            expectation_values=expectation_values,
            train_key=train_key,
        ),
        tune_config=tune.TuneConfig(
            search_alg=search_algo_instance,
            metric=prepended_metric,
            mode="min",
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=run_config,
    )

    results = tuner.fit()

    return results


def get_best_hypertuner_results(results, metric: LossMetric):
    prepended_metric = f"val/{metric}"

    with results.get_best_result(
        metric=prepended_metric, mode="min"
    ).checkpoint.as_directory() as checkpoint_dir:
        model_state, hist, data_config = load_model(checkpoint_dir, skip_history=False)
    return model_state, hist, data_config
