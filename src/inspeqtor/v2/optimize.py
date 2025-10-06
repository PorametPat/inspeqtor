import jax
import jax.numpy as jnp
import gpjax as gpx
from jax.scipy.stats import norm
from flax import struct
from inspeqtor.v2.control import ControlSequence, ravel_unravel_fn


def fit_gp(D: gpx.Dataset):
    kernel = gpx.kernels.RBF()  # 1-dimensional input
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    posterior = prior * likelihood

    opt_posterior, history = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),  # type: ignore
        train_data=D,
        trainable=gpx.parameters.Parameter,
        verbose=True,
    )
    return opt_posterior, history


def gp_predict(x, posterior: gpx.gps.ConjugatePosterior, D: gpx.Dataset):
    latent_dist = posterior.predict(x, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    # For the Gaussian process, only mean and variance should be enough?
    predictive_mean = predictive_dist.mean
    predictive_std = jnp.sqrt(predictive_dist.variance)
    return predictive_mean, predictive_std


def gaussian_process(x, D: gpx.Dataset):
    opt_posterior, _ = fit_gp(D)

    return gp_predict(x, opt_posterior, D)


def expected_improvement(
    y_best, posterior_mean, posterior_var, exploration_factor
) -> jnp.ndarray:
    # https://github.com/alonfnt/bayex/blob/main/bayex/acq.py
    std = jnp.sqrt(posterior_var)
    a = posterior_mean - y_best - exploration_factor
    z = a / std

    return a * norm.cdf(z) + std * norm.pdf(z)


@struct.dataclass
class BayesOptState:
    dataset: gpx.Dataset
    control: ControlSequence


def init_opt_state(x, y, control) -> BayesOptState:
    return BayesOptState(dataset=gpx.Dataset(X=x, y=y), control=control)


def sample(
    key: jnp.ndarray,
    opt_state: BayesOptState,
    sample_size: int = 1000,
    num_suggest: int = 1,
    exploration_factor: float = 0.0,
) -> jnp.ndarray:
    y = opt_state.dataset.y
    assert isinstance(y, jnp.ndarray)
    y_best = jnp.max(y)

    ravel_fn, unravel_fn = ravel_unravel_fn(opt_state.control)
    params = jax.vmap(opt_state.control.sample_params)(
        jax.random.split(key, sample_size)
    )
    # In shape of (sample_size, ctrl_feature)
    ravel_param = jax.vmap(ravel_fn)(params)

    mean, variance = gaussian_process(ravel_param, opt_state.dataset)

    ei = expected_improvement(
        y_best, mean, variance, exploration_factor=exploration_factor
    )

    selected_indice = jnp.argsort(ei, descending=True)[:num_suggest]

    return ravel_param[selected_indice]


def update_gp(opt_state: BayesOptState, x, y) -> BayesOptState:
    return BayesOptState(
        dataset=opt_state.dataset + gpx.Dataset(X=x, y=y), control=opt_state.control
    )
