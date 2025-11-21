import jax
import jax.numpy as jnp
import gpjax as gpx
from jax.scipy.stats import norm
from flax import struct
from inspeqtor.v2.control import ControlSequence, ravel_unravel_fn


def fit_gaussian_process(D: gpx.Dataset):
    """Fit the Gaussian process given an instance of Dataset

    Args:
        D (gpx.Dataset): The `gpx.Dataset` instance

    Returns:
        tuple[]: _description_
    """
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


def predict_with_gaussian_process(
    x, posterior: gpx.gps.ConjugatePosterior, D: gpx.Dataset
) -> tuple[jnp.ndarray, jnp.ndarray]:
    latent_dist = posterior.predict(x, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)
    # For the Gaussian process, only mean and variance should be enough?
    predictive_mean = predictive_dist.mean
    predictive_std = jnp.sqrt(predictive_dist.variance)
    return predictive_mean, predictive_std


def predict_mean_and_std(
    x: jnp.ndarray, D: gpx.Dataset
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict a Gaussian distribution to the given `x` using the dataset `D`

    Args:
        x (jnp.ndarray): The array of points to evaluate the gaussian process.
        D (gpx.Dataset): The dataset contain observation from the real process.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: The array of mean and standard deviation of the Gaussian process at ponits `x`.
    """
    opt_posterior, _ = fit_gaussian_process(D)

    return predict_with_gaussian_process(x, opt_posterior, D)


def expected_improvement(
    y_best: jnp.ndarray,
    posterior_mean: jnp.ndarray,
    posterior_var: jnp.ndarray,
    exploration_factor: float,
) -> jnp.ndarray:
    """The expected improvement calculated using posterior mean and variance of the gaussian process.
    The exploration factor can be adjust to balance between exploration and exploitation.


    Args:
        y_best (jnp.ndarray): The current maximum value of y
        posterior_mean (jnp.ndarray): The posterior mean of the gaussian process
        posterior_var (jnp.ndarray): The posterior variance of the gaussian process
        exploration_factor (float): The factor that balance between exploration and exploitation. Set to 0. to maximize exploitation.

    Returns:
        jnp.ndarray: The expeced improvement corresponding to the points given from array of the posterior.
    """
    # https://github.com/alonfnt/bayex/blob/main/bayex/acq.py
    std = jnp.sqrt(posterior_var)
    a = posterior_mean - y_best - exploration_factor
    z = a / std

    return a * norm.cdf(z) + std * norm.pdf(z)


@struct.dataclass
class BayesOptState:
    """The dataclass holding optimization state for the gaussian process."""

    dataset: gpx.Dataset
    control: ControlSequence


def init_opt_state(x, y, control) -> BayesOptState:
    """Function to intialize the optimizer

    Args:
        x (jnp.ndarray): The input arguments
        y (jnp.ndarray): The observation corresponding to the input `x`
        control (_type_): The intance of control sequence.

    Returns:
        BayesOptState: The state of optimizer.
    """
    return BayesOptState(dataset=gpx.Dataset(X=x, y=y), control=control)


def suggest_next_candidates(
    key: jnp.ndarray,
    opt_state: BayesOptState,
    sample_size: int = 1000,
    num_suggest: int = 1,
    exploration_factor: float = 0.0,
) -> jnp.ndarray:
    """Sample new candidates for experiment using expected improvement.

    Args:
        key (jnp.ndarray): The jax random key
        opt_state (BayesOptState): The current optimizer state
        sample_size (int, optional): The internal number of sample size. Defaults to 1000.
        num_suggest (int, optional): The number of suggestion for next experiment. Defaults to 1.
        exploration_factor (float, optional): The factor that balance between exploration and exploitation. Set to 0. to maximize exploitation. Defaults to 0.0.

    Returns:
        jnp.ndarray: The suggest data points to evalute in the experiment.
    """
    y = opt_state.dataset.y
    assert isinstance(y, jnp.ndarray)
    y_best = jnp.max(y)

    ravel_fn, unravel_fn = ravel_unravel_fn(opt_state.control.get_structure())
    params = jax.vmap(opt_state.control.sample_params)(
        jax.random.split(key, sample_size)
    )
    # In shape of (sample_size, ctrl_feature)
    ravel_param = jax.vmap(ravel_fn)(params)

    mean, variance = predict_mean_and_std(ravel_param, opt_state.dataset)

    ei = expected_improvement(
        y_best, mean, variance, exploration_factor=exploration_factor
    )

    selected_indice = jnp.argsort(ei, descending=True)[:num_suggest]

    return ravel_param[selected_indice]


def add_observations(opt_state: BayesOptState, x, y) -> BayesOptState:
    """Function to update the optimization state using new data points `x` and `y`

    Args:
        opt_state (BayesOptState): The current optimization state
        x (jnp.ndarray): The input arguments
        y (jnp.ndarray): The observation corresponding to the input `x`

    Returns:
        BayesOptState: The updated optimization state.
    """
    return BayesOptState(
        dataset=opt_state.dataset + gpx.Dataset(X=x, y=y), control=opt_state.control
    )
