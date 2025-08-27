from deprecated import deprecated
import jax
import jax.numpy as jnp
from collections import namedtuple
from numpyro.infer import svi as svilib  # type: ignore
import numpyro  # type: ignore
from numpyro import handlers
from numpyro.contrib.module import (
    random_flax_module,
    flax_module,
    nnx_module,
)  # type: ignore
import numpyro.util
import numpyro.distributions as dist  # type: ignore
from numpyro.infer import autoguide, Predictive
import typing
from functools import partial, reduce
from flax import linen as nn, nnx
from dataclasses import dataclass
import pathlib
from enum import StrEnum, auto
from numpyro.contrib.module import ParamShape
from copy import deepcopy

from .constant import (
    default_expectation_values_order,
)
from .model import (
    ModelData,
    get_predict_expectation_value,
    unitary_to_expvals as unitary_to_expvals,
    toggling_unitary_to_expvals as toggling_unitary_to_expvals,
    toggling_unitary_with_spam_to_expvals as toggling_unitary_with_spam_to_expvals,
    observable_to_expvals as observable_to_expvals,
)
from .data import save_pytree_to_json, load_pytree_from_json
from .utils import expectation_value_to_prob_minus, binary_to_eigenvalue


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


def make_flax_probabilistic_graybox_model(
    name: str,  # graybox
    base_model: nn.Module | nnx.Module,
    adapter_fn: typing.Callable[..., jnp.ndarray],
    prior: dict[str, dist.Distribution] | dist.Distribution = dist.Normal(0.0, 1.0),
    flax_module: typing.Callable = random_flax_module,
):
    if flax_module in [random_flax_module, random_nnx_module]:
        module = partial(flax_module, prior=prior)
    else:
        module = flax_module

    is_nnx = False
    if flax_module in [random_nnx_module, nnx_module]:
        is_nnx = not is_nnx

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

        _kwargs = {} if is_nnx else {"input_shape": control_parameters.shape}

        # Initialize BMLP model
        model = module(name, base_model, **_kwargs)

        # Predict from control parameters
        output = model(control_parameters)

        # With unitary and Wo, calculate expectation values
        expvals = adapter_fn(output, unitaries)

        return expvals

    return graybox_probabilistic_model


def make_flax_probabilistic_graybox_model_with_spam(
    name: str,  # graybox
    base_model: nn.Module,
    spam_model: typing.Callable,
    adapter_fn: typing.Callable[..., jnp.ndarray],
    prior: dict[str, dist.Distribution] | dist.Distribution = dist.Normal(0.0, 1.0),
):
    if flax_module in [random_flax_module, random_nnx_module]:
        module = partial(flax_module, prior=prior)
    else:
        module = flax_module

    is_nnx = False
    if flax_module in [random_nnx_module, nnx_module]:
        is_nnx = not is_nnx

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

        _kwargs = {} if is_nnx else {"input_shape": control_parameters.shape}

        # Initialize BMLP model
        model = module(name, base_model, **_kwargs)

        # Predict from control parameters
        model_output = model(control_parameters)

        spam_params = spam_model()

        output = {"model_params": model_output, "spam_params": spam_params}

        # With unitary and Wo, calculate expectation values
        expvals = adapter_fn(output, unitaries)

        return expvals

    return graybox_probabilistic_model


def _update_params(params, new_params, prior, prefix=""):
    """
    A helper to recursively set prior to new_params.

    Note:
        We copy the code from `numpyro.contrib` directly for now.
    """
    for name, item in params.items():
        # Parse int to str
        if isinstance(name, int):
            _name = str(name)
        else:
            _name = name
        flatten_name = ".".join([prefix, _name]) if prefix else _name
        if isinstance(item, dict):
            assert not isinstance(prior, dict) or flatten_name not in prior
            new_item = new_params[name]
            _update_params(item, new_item, prior, prefix=flatten_name)
        elif (not isinstance(prior, dict)) or flatten_name in prior:
            if isinstance(params[name], ParamShape):
                param_shape = params[name].shape
            else:
                param_shape = jnp.shape(params[name])
                params[name] = ParamShape(param_shape)
            if isinstance(prior, dict):
                d = prior[flatten_name]
            elif callable(prior) and not isinstance(prior, dist.Distribution):
                d = prior(flatten_name, param_shape)
            else:
                d = prior

            param_batch_shape = param_shape[: len(param_shape) - d.event_dim]  # type: ignore
            # XXX: here we set all dimensions of prior to event dimensions.
            new_params[name] = numpyro.sample(
                flatten_name,
                d.expand(param_batch_shape).to_event(),  # type: ignore
            )


def random_nnx_module(
    name,
    nn_module,
    prior,
):
    """A primitive to create a random :mod:`~flax.nnx` style neural network
    which can be used in MCMC samplers. The parameters of the neural network
    will be sampled from ``prior``.

    Note:
            We copy the code from `numpyro.contrib` directly for now.
    """

    nn = nnx_module(name, nn_module)

    apply_fn = nn.func
    params = nn.args[0]
    other_args = nn.args[1:]
    keywords = nn.keywords

    new_params = deepcopy(params)

    with numpyro.handlers.scope(prefix=name):
        _update_params(params, new_params, prior)

    return partial(apply_fn, new_params, *other_args, **keywords)


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
        typing.Any: Argument of Distribution if Distribution is provided.
    """
    if isinstance(x, dist.Distribution):
        return x.get_args()
    else:
        return x


def construct_normal_priors(posterior):
    """Construct a dict of Normal Distributions with posterior

    Args:
        posterior (typing.Any): Dict of Normal distribution arguments

    Returns:
        typing.Any: dict of Normal distributions
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


@deprecated("Use ModelData instead")
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
        posterior_model (typing.Any): probabilsitic model
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
    """Make the numberical PDF from given sample using histogram method

    Args:
        sample (jnp.ndarray): Sample to make PDF.
        bins (int): The number of interval bin.
        srange (tuple, optional): The range of the pdf. Defaults to (-1, 1).

    Returns:
        typing.Any: The approximated numerical PDF
    """
    density, bin_edges = jnp.histogram(sample, bins=bins, range=srange, density=True)
    dx = jnp.diff(bin_edges)
    p = density * dx
    return p


def safe_kl_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Calculate the KL divergence where the infinity is converted to zero.

    Args:
        p (jnp.ndarray): The left PDF
        q (jnp.ndarray): The right PDF

    Returns:
        jnp.ndarray: The KL divergence
    """
    return jnp.sum(jnp.nan_to_num(jax.scipy.special.rel_entr(p, q), posinf=0.0))


def kl_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Calculate the KL divergence

    Args:
        p (jnp.ndarray): The left PDF
        q (jnp.ndarray): The right PDF

    Returns:
        jnp.ndarray:  The KL divergence
    """
    return jnp.sum(jax.scipy.special.rel_entr(p, q))


def safe_jensenshannon_divergence(p: jnp.ndarray, q: jnp.ndarray):
    """Calculate Jensen-Shannon Divergnece using KL divergence.
    Implement following: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html

    Args:
        p (jnp.ndarray): The left PDF
        q (jnp.ndarray): The right PDF

    Returns:
        typing.Any: _description_
    """
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
    """Calculate the Jensen-Shannon Divergence from sample

    Args:
        sample_1 (jnp.ndarray): The left PDF
        sample_2 (jnp.ndarray): The right PDF

    Returns:
        jnp.ndarray: The Jensen-Shannon Divergence of p and q
    """
    merged_sample = jnp.concat([sample_1, sample_2])
    bins = int(2 * (sample_2.shape[0]) ** (1 / 3))
    srange = jnp.min(merged_sample), jnp.max(merged_sample)

    dis_1 = make_pdf(sample_1, bins=bins, srange=srange)
    dis_2 = make_pdf(sample_2, bins=bins, srange=srange)

    return jensenshannon_divergence_from_pdf(dis_1, dis_2)


def batched_matmul(x, w, b):
    """A specialized batched matrix multiplication of weight and input x, then add the bias.
    This function is intended to be used in `dense_layer`

    Args:
        x (jnp.ndarray): The input x
        w (jnp.ndarray): The weight to multiply to x
        b (jnp.ndarray): The bias

    Returns:
        jnp.ndarray: Output of the operations.
    """
    return jnp.einsum(x, (..., 0), w, (..., 0, 1), (..., 1)) + b


def get_trace(fn, key=jax.random.key(0)):
    """Convinent function to get a trace of the probabilistic model in numpyro.

    Args:
        fn (function): The probabilistic model in numpyro.
        key (jnp.ndarray, optional): The random key. Defaults to jax.random.key(0).
    """

    def inner(*args, **kwargs):
        return numpyro.handlers.trace(numpyro.handlers.seed(fn, key)).get_trace(
            *args, **kwargs
        )

    return inner


def default_priors_fn(param_name: str):
    """This is a default prior function for the `dense_layer`

    Args:
        param_name (str): The site name of the parameters, if end with `sigma` will return Log Normal distribution,
                          otherwise, return Normal distribution

    Returns:
        typing.Any: _description_
    """
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
    """A custom probabilistic dense layer for neural network model.
    This function intended to be used with `numpyro`

    Args:
        x (jnp.ndarray): The input x
        name (str): Site name of the layer
        in_features (int): The size of the feature.
        out_features (int): The desired size of the output feature.
        priors_fn (typing.Callable[[str], dist.Distribution], optional): The prior function to be used for initializing prior distribution. Defaults to default_priors_fn.

    Returns:
        typing.Any: Output of the layer given x.
    """
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
    """The initialization function for deterministic dense layer

    Args:
        params_name (str): The site name

    Raises:
        ValueError: Unsupport site name

    Returns:
        typing.Any: The function to be used for parameters init given site name.
    """
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
    """The deterministic dense layer, to be used with SVI optimizer.

    Args:
        x (typing.Any): The input feature
        name (str): The site name
        in_features (int): The size of the input features
        out_features (int): The desired size of the output features.
        batch_shape (tuple[int, ...], optional): The batch size of the x. Defaults to ().
        init_fn (typing.Any, optional): Initilization function of the model parameters. Defaults to init_default.

    Returns:
        typing.Any: The output of the layer given x.
    """
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


def make_posteriors_fn(key: jnp.ndarray, guide, params, num_samples=10000):
    """Make the posterior distribution function that will
    return the posterior of parameter of the given name, from guide and parameters.

    Args:
        guide (typing.Any): The guide function
        params (typing.Any): The parameters of the guide
        num_samples (int, optional): The sample size. Defaults to 10000.

    Returns:
        typing.Any: A function of parameter name that return the sample from the posterior distribution of the parameters.
    """
    posterior_distribution = Predictive(
        model=guide, params=params, num_samples=num_samples
    )(key)

    posterior_dict = construct_normal_prior_from_samples(posterior_distribution)

    def posteriors_fn(param_name: str):
        return posterior_dict[param_name]

    return posteriors_fn


def auto_diagonal_normal_guide(
    model,
    *args,
    block_sample: bool = False,
    init_loc_fn=jnp.zeros,
    key: jnp.ndarray = jax.random.key(0),
):
    """Automatically generate guide from given model. Expected to be initialized with the example input of the model.
    The given input should also including the observed site.
    The blocking capability is intended to be used in the when the guide will be used with its corresponding model in anothe model.
    This is the avoid site name duplication, while allows for model to use newly sample from the guide.

    Args:
        model (typing.Any): The probabilistic model.
        block_sample (bool, optional): Flag to block the sample site. Defaults to False.
        init_loc_fn (typing.Any, optional): Initialization of guide parameters function. Defaults to jnp.zeros.

    Returns:
        typing.Any: _description_
    """
    model_trace = handlers.trace(handlers.seed(model, key)).get_trace(*args)
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


def init_normal_dist_fn(name: str):
    if name.startswith("SPAM/"):
        return partial(dist.TruncatedNormal, low=0.0, high=1.0)

    return dist.Normal


def init_params_fn(name: str, shape: tuple[int, ...]):
    if name.endswith("_loc"):
        constraint = (
            dist.constraints.unit_interval
            if name.startswith("SPAM/")
            else dist.constraints.real
        )
        return numpyro.param(name, 0.5 * jnp.ones(shape), constraint=constraint)
    elif name.endswith("_scale"):
        return numpyro.param(
            name, 0.1 * jnp.ones(shape), constraint=dist.constraints.softplus_positive
        )
    else:
        raise ValueError(f"name: {name} is not supported")


def auto_diagonal_normal_guide_v2(
    model,
    *args,
    init_dist_fn=init_normal_dist_fn,
    init_params_fn=init_params_fn,
    block_sample: bool = False,
    key: jnp.ndarray = jax.random.key(0),
):
    """Automatically generate guide from given model. Expected to be initialized with the example input of the model.
    The given input should also including the observed site.
    The blocking capability is intended to be used in the when the guide will be used with its corresponding model in anothe model.
    This is the avoid site name duplication, while allows for model to use newly sample from the guide.

    Args:
        model (typing.Any): The probabilistic model.
        block_sample (bool, optional): Flag to block the sample site. Defaults to False.
        init_loc_fn (typing.Any, optional): Initialization of guide parameters function. Defaults to jnp.zeros.

    Returns:
        typing.Any: _description_
    """
    # get the trace of the model
    model_trace = handlers.trace(handlers.seed(model, key)).get_trace(*args)
    # Then get only the sample site with observed equal to false
    sample_sites = [v for k, v in model_trace.items() if v["type"] == "sample"]
    non_observed_sites = [v for v in sample_sites if not v["is_observed"]]
    params_sites = [
        {"name": v["name"], "shape": v["value"].shape} for v in non_observed_sites
    ]

    def sample_fn(
        params_loc: dict[str, typing.Any], params_scale: dict[str, typing.Any]
    ):
        samples = {}
        # Sample from Normal distribution
        for (k_loc, v_loc), (k_scale, v_scale) in zip(
            params_loc.items(), params_scale.items(), strict=True
        ):
            s = numpyro.sample(
                k_loc,
                init_dist_fn(k_loc)(v_loc, v_scale).to_event(),  # type: ignore
            )
            samples[k_loc] = s

        return samples

    def guide(
        *args,
        **kwargs,
    ):
        params_loc = {
            param["name"]: init_params_fn(f"{param['name']}_loc", param["shape"])
            for param in params_sites
        }

        params_scale = {
            param["name"]: init_params_fn(f"{param['name']}_scale", param["shape"])
            for param in params_sites
        }

        if block_sample:
            with handlers.block():
                samples = sample_fn(params_loc, params_scale)
        else:
            samples = sample_fn(params_loc, params_scale)

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


class SVIResult(ModelData):
    pass


def compose(functions):
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def make_predictive_model_fn_v2(
    model,
    guide,
    params,
    shots: int,
):
    """Make a postirior predictive model function from model, guide, SVI parameters, and the number of shots.

    Args:
        model (typing.Any): Probabilistic model.
        guide (typing.Any): Gudie corresponded to the model
        params (typing.Any): SVI parameters of the guide
        shots (int): The number of shots

    Returns:
        typing.Any: The posterior predictive model.
    """
    predictive = Predictive(
        model, guide=guide, params=params, num_samples=shots, return_sites=["obs"]
    )

    def predictive_fn(*args, **kwargs):
        return predictive(*args, **kwargs)["obs"]

    return predictive_fn


def make_predictive_SGM_model(
    model: nn.Module, model_params, output_to_expectation_values_fn, shots: int
):
    """Make a predictive model from given SGM model, the model parameters, and number of shots.

    Args:
        model (nn.Module): Flax model
        model_params (typing.Any): The model parameters.
        shots (int): The number of shots.
    """

    def predictive_model(
        key: jnp.ndarray, control_param: jnp.ndarray, unitaries: jnp.ndarray
    ):
        output = model.apply(model_params, control_param)
        predicted_expvals = output_to_expectation_values_fn(output, unitaries)

        return binary_to_eigenvalue(
            jax.vmap(jax.random.bernoulli, in_axes=(0, None))(
                jax.random.split(key, shots),
                expectation_value_to_prob_minus(predicted_expvals),
            ).astype(jnp.int_)
        ).mean(axis=0)

    return predictive_model


def make_predictive_MCDGM_model(model: nn.Module, model_params):
    """Make a predictive model from given Monte-Carlo Dropout Graybox model, and the model parameters.

    Args:
        model (nn.Module): Monte-Carlo Dropout Graybox model
        model_params (typing.Any): The model parameters
    """

    def predictive_model(
        key: jnp.ndarray, control_param: jnp.ndarray, unitaries: jnp.ndarray
    ):
        wo_params = model.apply(
            model_params,
            control_param,
            rngs={"dropout": key},
        )

        predicted_expvals = get_predict_expectation_value(
            wo_params,  # type: ignore
            unitaries,
            default_expectation_values_order,
        )

        return predicted_expvals

    return predictive_model


def make_predictive_resampling_model(
    predictive_fn: typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], shots: int
) -> typing.Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Make a binary predictive model from given SGM model, the model parameters, and number of shots.

    Args:
        predictive_fn (typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]): The predictive_fn embeded with the SGM model.
        shots (int): The number of shots.

    Returns:
        typing.Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]: Binary predictive model.
    """

    def predictive_model(
        key: jnp.ndarray, control_parameters: jnp.ndarray, unitaries: jnp.ndarray
    ):
        predicted_expvals = predictive_fn(control_parameters, unitaries)

        return binary_to_eigenvalue(
            jax.vmap(jax.random.bernoulli, in_axes=(0, None))(
                jax.random.split(key, shots),
                expectation_value_to_prob_minus(predicted_expvals),
            ).astype(jnp.int_)
        ).mean(axis=0)

    return predictive_model
