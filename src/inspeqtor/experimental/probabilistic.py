import jax
import jax.numpy as jnp
from collections import namedtuple
from numpyro.infer import svi as svilib  # type: ignore
import numpyro  # type: ignore
from numpyro import handlers
from numpyro.contrib.module import random_flax_module, flax_module  # type: ignore
import numpyro.distributions as dist  # type: ignore
from numpyro.infer import autoguide
import typing
from functools import partial
from flax import linen as nn
from dataclasses import dataclass
import pathlib
import json
from enum import StrEnum, auto

from .constant import default_expectation_values_order, X, Y, Z
from .model import get_predict_expectation_value, unitary


def expectation_value_to_prob_plus(expectation_value: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the probability of -1 and 1 for the given expectation value
    E[O] = -1 * P[O = -1] + 1 * P[O = 1], where P[O = -1] + P[O = 1] = 1
    Thus, E[O] = -1 * (1 - P[O = 1]) + 1 * P[O = 1]
    E[O] = 2 * P[O = 1] - 1 -> P[O = 1] = (E[O] + 1) / 2
    Args:
        expectation_value (jnp.ndarray): Expectation value of quantum observable

    Returns:
        jnp.ndarray: Probability of measuring plus eigenvector
    """

    return (expectation_value + 1) / 2


def expectation_value_to_prob_minus(expectation_value: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the probability of -1 and 1 for the given expectation value
    E[O] = -1 * P[O = -1] + 1 * P[O = 1], where P[O = -1] + P[O = 1] = 1
    Thus, E[O] = -1 * P[O = -1] + 1 * (1 - P[O = -1])
    E[O] = 1 - 2 * P[O = -1] -> P[O = -1] = (1 - E[O]) / 2
    Args:
        expectation_value (jnp.ndarray): Expectation value of quantum observable

    Returns:
        jnp.ndarray: Probability of measuring minus eigenvector
    """

    return (1 - expectation_value) / 2


def expectation_value_to_eigenvalue(
    expectation_value: jnp.ndarray, SHOTS: int
) -> jnp.ndarray:
    """Convert expectation value to eigenvalue

    Args:
        expectation_value (jnp.ndarray): Expectation value of quantum observable
        SHOTS (int): The number of shots used to produce expectation value

    Returns:
        jnp.ndarray: Array of eigenvalues
    """
    return jnp.where(
        jnp.broadcast_to(jnp.arange(SHOTS), expectation_value.shape + (SHOTS,))
        < jnp.around(
            expectation_value_to_prob_plus(
                jnp.reshape(expectation_value, expectation_value.shape + (1,))
            )
            * SHOTS
        ).astype(jnp.int32),
        1,
        -1,
    ).astype(jnp.int32)


def eigenvalue_to_binary(eigenvalue: jnp.ndarray) -> jnp.ndarray:
    """Convert -1 to 1, and 0 to 1
    This implementation should be differentiable

    Args:
        eigenvalue (jnp.ndarray): Eigenvalue to convert to bit value

    Returns:
        jnp.ndarray: Binary array
    """

    return (-1 * eigenvalue + 1) / 2


def binary_to_eigenvalue(binary: jnp.ndarray) -> jnp.ndarray:
    """Convert 1 to -1, and 0 to 1
    This implementation should be differentiable

    Args:
        binary (jnp.ndarray): Bit value to convert to eigenvalue

    Returns:
        jnp.ndarray: Eigenvalue array
    """

    return -1 * (binary * 2 - 1)


def make_update_fn(
    svi: svilib.SVI,
    **kwargs,
):
    def update_fn(svi_state: svilib.SVIState):
        return svi.update(svi_state, **kwargs)

    return update_fn


def make_evaluate_fn(
    svi: svilib.SVI,
    **kwargs,
):
    def evaluate_fn(svi_state: svilib.SVIState):
        return svi.evaluate(svi_state, **kwargs)

    return evaluate_fn


SVIRunResult = namedtuple(
    "SVIRunResult",
    ["params", "state", "losses", "eval_losses"],
)


def unitary_model_prediction_to_expvals(output, unitaries: jnp.ndarray) -> jnp.ndarray:
    """Function to be used with Unitary-based model for probabilistic model construction
    with `make_probabilistic_model` function.

    Args:
        output (_type_): The output from Unitary-based model
        unitaries (jnp.ndarray): Ideal unitary, ignore in this function.

    Returns:
        jnp.ndarray: Expectation values array
    """
    U = unitary(output)
    return get_predict_expectation_value(
        {"X": X, "Y": Y, "Z": Z},
        U,
        default_expectation_values_order,
    )


def wo_model_prediction_to_expvals(output, unitaries: jnp.ndarray) -> jnp.ndarray:
    """Function to be used with Wo-based model for probabilistic model construction
    with `make_probabilistic_model` function.

    Args:
        output (_type_): The output from Wo-based model
        unitaries (jnp.ndarray): Ideal unitary, ignore in this function.

    Returns:
        jnp.ndarray: Expectation values array
    """
    return get_predict_expectation_value(
        Wos=output,
        unitaries=unitaries,
        evaluate_expectation_values=default_expectation_values_order,
    )


def make_probabilistic_model(
    base_model: nn.Module,
    model_prediction_to_expvals_fn: typing.Callable[..., jnp.ndarray],
    bnn_prior: dict[str, dist.Distribution] | dist.Distribution = dist.Normal(0.0, 1.0),
    shots: int = 1,
    block_graybox: bool = False,
    enable_bnn: bool = True,
    separate_observables: bool = False,
):
    """Make probabilistic model from the Statistical model with priors

    Args:
        base_model (nn.Module): The statistical based model, currently only support flax.linen module
        model_prediction_to_expvals_fn (typing.Callable[..., jnp.ndarray]): Function to convert output from model to expectation values array
        bnn_prior (dict[str, dist.Distribution] | dist.Distribution, optional): The priors of BNN. Defaults to dist.Normal(0.0, 1.0).
        shots (int, optional): The number of shots forcing PGM to sample. Defaults to 1.
        block_graybox (bool, optional): If true, the latent variables in Graybox model will be hidden, i.e. not traced by `numpyro`. Defaults to False.
        enable_bnn (bool, optional): If true, the statistical model will be convert to probabilistic model. Defaults to True.
        separate_observables (bool, optional): If true, the observable will be separate into dict form. Defaults to False.

    Returns:
        typing.Callable: Probabilistic Graybox Model
    """
    module = partial(random_flax_module, prior=bnn_prior) if enable_bnn else flax_module

    def graybox_probabilistic_model(
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
    ):
        """Graybox model
        Args:
            control_parameters (jnp.ndarray): control parameters
            unitaries (jnp.ndarray): The unitary according to the control parameters

        Returns:
            jnp.ndarray: Expectation values
        """

        samples_shape = control_parameters.shape[:-2]
        unitaries = jnp.broadcast_to(unitaries, samples_shape + unitaries.shape[-3:])

        # Initialize BMLP model
        model = module(
            "graybox",
            base_model,
            input_shape=control_parameters.shape,
        )

        # Predict from control parameters
        output = model(control_parameters)

        # With unitary and Wo, calculate expectation values
        expvals = model_prediction_to_expvals_fn(output, unitaries)

        numpyro.deterministic("expectation_values", expvals)

        return expvals

    def block_graybox_fn(
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
    ):
        key = numpyro.prng_key()
        with handlers.block(), handlers.seed(rng_seed=key):
            expvals = graybox_probabilistic_model(control_parameters, unitaries)

        return expvals

    graybox_fn = block_graybox_fn if block_graybox else graybox_probabilistic_model

    def bernoulli_model(
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        observables: jnp.ndarray | None = None,
    ):
        expvals = graybox_fn(control_parameters, unitaries)

        if observables is None:
            sizes = control_parameters.shape[:-1] + (18,)
            if shots > 1:
                sizes = (shots,) + sizes
        else:
            sizes = observables.shape

        if separate_observables:
            expvals_samples = {}

            for idx, exp in enumerate(default_expectation_values_order):
                s = numpyro.sample(
                    f"obs/{exp.initial_state}/{exp.observable}",
                    dist.BernoulliProbs(
                        probs=expectation_value_to_prob_minus(expvals[..., idx])
                    ),
                    obs=(observables[..., idx] if observables is not None else None),
                )

                expvals_samples[f"obs/{exp.initial_state}/{exp.observable}"] = s

        else:
            with numpyro.plate_stack("plate", sizes=list(sizes)):
                expvals_samples = numpyro.sample(
                    "obs",
                    dist.BernoulliProbs(probs=expectation_value_to_prob_minus(expvals)),
                    obs=observables,
                    infer={"enumerate": "parallel"},
                )

        return expvals_samples

    return bernoulli_model


def save_pytree_to_json(pytree, path: str | pathlib.Path):
    """Save given pytree to json file, the path must end with extension of .json

    Args:
        pytree (_type_): The pytree to save
        path (str | pathlib.Path): File path to save

    """

    data = jax.tree.map(
        lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, pytree
    )

    if isinstance(path, str):
        path = pathlib.Path(path)

    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_pytree_from_json(path: str | pathlib.Path, array_keys: list[str] = []):
    """Load pytree from json

    Args:
        path (str | pathlib.Path): Path to JSON file containing pytree
        array_keys (list[str], optional): list of key to convert to jnp.numpy. Defaults to [].

    Raises:
        ValueError: Provided path is not point to .json file

    Returns:
        _type_: Pytree loaded from JSON
    """

    # Validate that file extension is .json
    extension = str(path).split(".")[-1]

    if extension != "json":
        raise ValueError("File extension must be json")

    if isinstance(path, str):
        path = pathlib.Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    def is_leaf(x):
        return isinstance(x, list)

    temp_data = {}
    for key, value in data.items():
        if key in array_keys:
            temp_data[key] = jax.tree.map(
                lambda x: jnp.array(x), value, is_leaf=is_leaf
            )
        else:
            temp_data[key] = value

    return temp_data


def get_args_of_distribution(x):
    """Get the arguments used to construct Distribution, if the provided parameter is not Distribution, return it.
    So that the function can be used with `jax.tree.map`.

    Args:
        x (typing.Any): Maybe Distribution

    Returns:
        _type_: Argument of Distribution if Distribution is provided.
    """
    if isinstance(x, dist.Distribution):
        return x.get_args()
    else:
        return x


def construct_normal_priors(posterior):
    """Construct a dict of Normal Distributions with posterior

    Args:
        posterior (_type_): Dict of Normal distribution arguments

    Returns:
        _type_: dict of Normal distributions
    """
    posterior_distributions = {}
    assert isinstance(posterior, dict)
    for name, value in posterior.items():
        assert isinstance(name, str)
        assert isinstance(value, dict)
        posterior_distributions[name] = dist.Normal(value["loc"], value["scale"])  # type: ignore
    return posterior_distributions


def construct_normal_prior_from_samples(
    posterior_samples: dict[str, jnp.ndarray],
) -> dict[str, dist.Distribution]:
    """Construct a dict of Normal Distributions with posterior sample

    Args:
        posterior_samples (dict[str, jnp.ndarray]): Posterior sample

    Returns:
        dict[str, dist.Distribution]: dict of Normal Distributions
    """

    posterior_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), posterior_samples)
    posterior_std = jax.tree.map(lambda x: jnp.std(x, axis=0), posterior_samples)

    prior = {}
    for name, mean in posterior_mean.items():
        prior[name] = dist.Normal(mean, posterior_std[name])

    return prior


@dataclass
class ProbabilisticModel:
    """Dataclass to save and load probabilistic model from inference result and file."""

    posterior: dict[str, jnp.ndarray]
    shots: int
    hidden_sizes: list[list[int]]

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "ProbabilisticModel":
        data = load_pytree_from_json(path, array_keys=["posterior"])

        return cls(
            posterior=data["posterior"],
            shots=data["shots"],
            hidden_sizes=data["hidden_sizes"],
        )

    @classmethod
    def from_result(
        cls,
        guide: autoguide.AutoDiagonalNormal,
        svi_params,
        key: jnp.ndarray,
        shots: int,
        hidden_sizes: list[list[int]],
        sample_shape: tuple[int, ...] = (10000,),
        prefix: str = "graybox/",
    ):
        posterior_samples = guide.sample_posterior(
            key, svi_params, sample_shape=sample_shape
        )

        posterior = construct_normal_prior_from_samples(posterior_samples)
        posterior = {
            name.split("/")[1]: prior
            for name, prior in posterior.items()
            if name.startswith(prefix)
        }

        posterior = jax.tree.map(
            get_args_of_distribution,
            posterior,
            is_leaf=lambda x: isinstance(x, dist.Distribution),
        )

        return cls(
            posterior=posterior,
            shots=shots,
            hidden_sizes=hidden_sizes,
        )

    def to_file(self, path: str | pathlib.Path):
        data = {
            "posterior": self.posterior,
            "shots": self.shots,
            "hidden_sizes": self.hidden_sizes,
        }

        save_pytree_to_json(data, path)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n\thidden_sizes={str(self.hidden_sizes)}\n\tshots={self.shots}\n\tposterior=...\n)"


class LearningModel(StrEnum):
    """The learning model."""

    TruncatedNormal = auto()
    BernoulliProbs = auto()


def make_predictive_fn(
    posterior_model,
    learning_model: LearningModel,
):
    """Construct predictive model from the probabilsitic model.

    Args:
        posterior_model (_type_): probabilsitic model
        learning_model (LearningModel): _description_
    """

    def binary_predict_expectation_values(
        key: jnp.ndarray,
        control_params: jnp.ndarray,
        unitary: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.mean(
            binary_to_eigenvalue(
                handlers.seed(posterior_model, key)(  # type: ignore
                    control_params, unitary
                )
            ),
            axis=0,
        )

    def normal_predict_expectation_values(
        key: jnp.ndarray,
        control_params: jnp.ndarray,
        unitary: jnp.ndarray,
    ) -> jnp.ndarray:
        return handlers.seed(posterior_model, key)(  # type: ignore
            control_params, unitary
        )

    return (
        binary_predict_expectation_values
        if learning_model == LearningModel.BernoulliProbs
        else normal_predict_expectation_values
    )
