from deprecated import deprecated
import jax
import jax.numpy as jnp
from collections import namedtuple
from numpyro.infer import svi as svilib  # type: ignore
import numpyro  # type: ignore
from numpyro import handlers
from numpyro.contrib.module import random_flax_module, flax_module  # type: ignore
import numpyro.util
import numpyro.distributions as dist  # type: ignore
from numpyro.infer import autoguide, Predictive
import typing
from functools import partial, reduce
from flax import linen as nn
from dataclasses import dataclass
import pathlib
import json
from enum import StrEnum, auto
from numpyro.contrib.module import ParamShape
import chex

from .constant import default_expectation_values_order, X, Y, Z
from .model import get_predict_expectation_value, unitary
from .data import save_pytree_to_json, load_pytree_from_json


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


def make_flax_probabilistic_graybox_model(
    name: str,  # graybox
    base_model: nn.Module,
    model_prediction_to_expvals_fn: typing.Callable[..., jnp.ndarray],
    prior: dict[str, dist.Distribution] | dist.Distribution = dist.Normal(0.0, 1.0),
    enable_bnn: bool = True,
):
    module = partial(random_flax_module, prior=prior) if enable_bnn else flax_module

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
            name,
            base_model,
            input_shape=control_parameters.shape,
        )

        # Predict from control parameters
        output = model(control_parameters)

        # With unitary and Wo, calculate expectation values
        expvals = model_prediction_to_expvals_fn(output, unitaries)

        return expvals

    return graybox_probabilistic_model


def make_probabilistic_model(
    graybox_probabilistic_model: typing.Callable[..., jnp.ndarray],
    shots: int = 1,
    block_graybox: bool = False,
    separate_observables: bool = False,
    log_expectation_values: bool = False,
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

        if log_expectation_values:
            numpyro.deterministic("expectation_values", expvals)

        if observables is None:
            sizes = control_parameters.shape[:-1] + (18,)
            if shots > 1:
                sizes = (shots,) + sizes
        else:
            sizes = observables.shape

        # The plate is for the shots prediction to work properly
        with numpyro.util.optional(
            shots > 1, numpyro.plate_stack("plate", sizes=list(sizes)[:-1])
        ):
            if separate_observables:
                expvals_samples = {}

                for idx, exp in enumerate(default_expectation_values_order):
                    s = numpyro.sample(
                        f"obs/{exp.initial_state}/{exp.observable}",
                        dist.BernoulliProbs(
                            probs=expectation_value_to_prob_minus(
                                jnp.expand_dims(expvals[..., idx], axis=-1)
                            )
                        ).to_event(1),
                        obs=(
                            observables[..., idx] if observables is not None else None
                        ),
                    )

                    expvals_samples[f"obs/{exp.initial_state}/{exp.observable}"] = s

            else:
                expvals_samples = numpyro.sample(
                    "obs",
                    dist.BernoulliProbs(
                        probs=expectation_value_to_prob_minus(expvals)
                    ).to_event(1),
                    obs=observables,
                    infer={"enumerate": "parallel"},
                )

        return expvals_samples

    return bernoulli_model


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
        # data = load_pytree_from_json(path, array_keys=["posterior"])
        data = load_pytree_from_json(path)

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


def make_pdf(sample: jnp.ndarray, bins: int, srange=(-1, 1)):
    density, bin_edges = jnp.histogram(sample, bins=bins, range=srange, density=True)
    dx = jnp.diff(bin_edges)
    p = density * dx
    return p


def safe_kl_divergence(p: jnp.ndarray, q: jnp.ndarray):
    return jnp.sum(jnp.nan_to_num(jax.scipy.special.rel_entr(p, q), posinf=0.0))


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray):
    return jnp.sum(jax.scipy.special.rel_entr(p, q))


def safe_jensenshannon_divergence(p: jnp.ndarray, q: jnp.ndarray):
    # Implement following: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    # Compute pointwise mean of p and q
    m = (p + q) / 2
    return (safe_kl_divergence(p, m) + safe_kl_divergence(q, m)) / 2


def jensenshannon_divergence_from_pdf(p: jnp.ndarray, q: jnp.ndarray):
    """Calculate the Jensen-Shannon Divergence from PMF

    Example
    ```python
    key = jax.random.key(0)
    key_1, key_2 = jax.random.split(key)
    sample_1 = jax.random.normal(key_1, shape=(10000, ))
    sample_2 = jax.random.normal(key_2, shape=(10000, ))

    # Determine srange from sample
    merged_sample = jnp.concat([sample_1, sample_2])
    srange = jnp.min(merged_sample), jnp.max(merged_sample)

    # https://stats.stackexchange.com/questions/510699/discrete-kl-divergence-with-decreasing-bin-width
    # Recommend this book: https://catalog.lib.uchicago.edu/vufind/Record/6093380/TOC
    bins = int(2 * (sample_2.shape[0]) ** (1/3))
    # bins = 10
    dis_1 = sq.probabilistic.make_pdf(sample_1, bins=bins, srange=srange)
    dis_2 = sq.probabilistic.make_pdf(sample_2, bins=bins, srange=srange)

    jsd = sq.probabilistic.jensenshannon_divergence_from_pdf(dis_1, dis_2)

    ```

    Args:
        p (jnp.ndarray): The 1st probability mass function
        q (jnp.ndarray): The 1st probability mass function

    Returns:
        jnp.ndarray: The Jensen-Shannon Divergence of p and q
    """
    # Note for JSD: https://medium.com/data-science/how-to-understand-and-use-jensen-shannon-divergence-b10e11b03fd6
    # Implement following: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    # Compute pointwise mean of p and q
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2


def jensenshannon_divergence_from_sample(
    sample_1: jnp.ndarray, sample_2: jnp.ndarray
) -> jnp.ndarray:
    merged_sample = jnp.concat([sample_1, sample_2])
    bins = int(2 * (sample_2.shape[0]) ** (1 / 3))
    srange = jnp.min(merged_sample), jnp.max(merged_sample)

    dis_1 = make_pdf(sample_1, bins=bins, srange=srange)
    dis_2 = make_pdf(sample_2, bins=bins, srange=srange)

    return jensenshannon_divergence_from_pdf(dis_1, dis_2)


def batched_matmul(x, w, b):
    return jnp.einsum(x, (..., 0), w, (..., 0, 1), (..., 1)) + b


def get_trace(fn, key=jax.random.key(0)):
    def inner(*args, **kwargs):
        return numpyro.handlers.trace(numpyro.handlers.seed(fn, key)).get_trace(
            *args, **kwargs
        )

    return inner


def default_priors_fn(param_name: str):
    if param_name.endswith("sigma"):
        return dist.LogNormal(0, 1)

    return dist.Normal(0, 1)


def dense_layer(
    x: jnp.ndarray,
    name: str,
    in_features: int,
    out_features: int,
    priors_fn: typing.Callable[[str], dist.Distribution] = default_priors_fn,
):
    w_name = f"{name}.kernel"
    w = numpyro.sample(
        w_name,
        priors_fn(w_name).expand((in_features, out_features)).to_event(2),  # type: ignore
    )
    b_name = f"{name}.bias"
    b = numpyro.sample(
        b_name,
        priors_fn(b_name).expand((out_features,)).to_event(1),  # type: ignore
    )
    return batched_matmul(x, w, b)  # type: ignore


def init_default(params_name: str):
    if params_name.endswith("kernel"):
        return jnp.ones
    elif params_name.endswith("bias"):
        return lambda x: 0.1 * jnp.ones(x)
    else:
        raise ValueError("Unsupport param name")


def dense_deterministic_layer(
    x,
    name: str,
    in_features: int,
    out_features: int,
    batch_shape: tuple[int, ...] = (),
    init_fn=init_default,
):
    # Sample weights - shape (in_features, out_features)
    weight_shape = batch_shape + (in_features, out_features)
    W_name = f"{name}.kernel"
    W = numpyro.param(
        W_name,
        init_fn(W_name)(shape=weight_shape),  # type: ignore
    )

    # Sample bias - shape (out_features,)
    bias_shape = batch_shape + (out_features,)
    b_name = f"{name}.bias"
    b = numpyro.param(b_name, init_fn(b_name)(shape=bias_shape))  # type: ignore

    return batched_matmul(x, W, b)  # type: ignore


def make_posteriors_fn(guide, params, num_samples=10000):
    posterior_distribution = Predictive(
        model=guide, params=params, num_samples=num_samples
    )(jax.random.key(0))

    posterior_dict = construct_normal_prior_from_samples(posterior_distribution)

    def posteriors_fn(param_name: str):
        return posterior_dict[param_name]

    return posteriors_fn


def auto_diagonal_noraml_guide(
    model, *args, block_sample: bool = False, init_loc_fn=jnp.zeros
):
    model_trace = handlers.trace(handlers.seed(model, jax.random.key(0))).get_trace(
        *args
    )
    # get the trace of the model
    # Then get only the sample site with observed equal to false
    sample_sites = [v for k, v in model_trace.items() if v["type"] == "sample"]
    non_observed_sites = [v for v in sample_sites if not v["is_observed"]]
    params_sites = [
        {"name": v["name"], "shape": v["value"].shape} for v in non_observed_sites
    ]

    def guide(
        *args,
        **kwargs,
    ):
        params_loc = {
            param["name"]: numpyro.param(
                f"{param['name']}_loc", init_loc_fn(param["shape"])
            )
            for param in params_sites
        }

        params_scale = {
            param["name"]: numpyro.param(
                f"{param['name']}_scale",
                0.1 * jnp.ones(param["shape"]),
                constraint=dist.constraints.softplus_positive,
            )
            for param in params_sites
        }

        samples = {}

        if block_sample:
            with handlers.block():
                # Sample from Normal distribution
                for (k_loc, v_loc), (k_scale, v_scale) in zip(
                    params_loc.items(), params_scale.items(), strict=True
                ):
                    s = numpyro.sample(
                        k_loc,
                        dist.Normal(v_loc, v_scale).to_event(),  # type: ignore
                    )
                    samples[k_loc] = s
        else:
            # Sample from Normal distribution
            for (k_loc, v_loc), (k_scale, v_scale) in zip(
                params_loc.items(), params_scale.items(), strict=True
            ):
                s = numpyro.sample(
                    k_loc,
                    dist.Normal(v_loc, v_scale).to_event(),  # type: ignore
                )
                samples[k_loc] = s

        return samples

    return guide


def is_leaf_array(x):
    return isinstance(x, list)


def to_array(x):
    return jnp.array(x) if is_leaf_array(x) else x


def is_dict_paramshape(x):
    if isinstance(x, dict) and len(x) == 1 and "shape" in x:
        return True

    return False


def parse_param_shape(x):
    if is_dict_paramshape(x):
        return ParamShape(shape=tuple(x["shape"]))
    else:
        return x


@deprecated
def load_pytree_from_json_old(path: str | pathlib.Path, array_keys: list[str] = []):
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

    # def is_leaf(x):
    #     return isinstance(x, list)

    temp_data = {}
    for key, value in data.items():
        if key in array_keys:
            temp_data[key] = jax.tree.map(
                parse_param_shape, value, is_leaf=is_dict_paramshape
            )

            # Parse list to jnp.ndarray
            temp_data[key] = jax.tree.map(
                to_array, temp_data[key], is_leaf=is_leaf_array
            )
        else:
            temp_data[key] = value

    return temp_data


@dataclass
class SVIResult:
    params: chex.ArrayTree
    config: dict[str, typing.Any]

    def to_file(self, path: str | pathlib.Path):
        data = {
            "params": self.params,
            "config": self.config,
        }

        save_pytree_to_json(data, path)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "SVIResult":
        data = load_pytree_from_json(path)

        return cls(
            params=data["params"],
            config=data["config"],
        )

    def __eq__(self, value):
        if not isinstance(value, type(self)):
            raise ValueError("The compared value is not SVIResult object")

        try:
            chex.assert_trees_all_close(self.params, value.params)
        except AssertionError:
            return False

        return True if value.config == self.config else False


def compose(functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


DataBundled = namedtuple(
    "DataBundled", ["control_params", "unitaries", "observables", "aux"]
)


def make_predictive_model_fn_v2(
    model,
    guide,
    params,
    shots,
):
    predictive = Predictive(
        model, guide=guide, params=params, num_samples=shots, return_sites=["obs"]
    )

    def predictive_fn(*args, **kwargs):
        return predictive(*args, **kwargs)["obs"]

    return predictive_fn


def make_predictive_SGM_model(model, model_params, shots: int):
    def predictive_model(
        key: jnp.ndarray, control_param: jnp.ndarray, unitaries: jnp.ndarray
    ):
        wo_params = model.apply(
            model_params,
            control_param,
        )

        predicted_expvals = get_predict_expectation_value(
            wo_params,
            unitaries,
            default_expectation_values_order,
        )
        return binary_to_eigenvalue(
            jax.vmap(jax.random.bernoulli, in_axes=(0, None))(
                jax.random.split(key, shots),
                expectation_value_to_prob_minus(predicted_expvals),
            ).astype(jnp.int_)
        ).mean(axis=0)

    return predictive_model


def make_predictive_MCDG_model(model, model_params):
    def predictive_model(
        key: jnp.ndarray, control_param: jnp.ndarray, unitaries: jnp.ndarray
    ):
        wo_params = model.apply(
            model_params,
            control_param,
            rngs={"dropout": key},
        )

        predicted_expvals = get_predict_expectation_value(
            wo_params,
            unitaries,
            default_expectation_values_order,
        )

        return predicted_expvals

    return predictive_model
