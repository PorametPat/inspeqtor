import jax
import jax.numpy as jnp
from flax import linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module, flax_module


def generate_data(a=4.0, s=0.25):
    # Parameters

    # True function
    def y_true(x):
        return (x - a) ** 2

    # Noisy function
    def y_noise(x, key):
        return y_true(x) + s * jax.random.normal(key, x.shape)

    return y_true, y_noise


def generate_data_v2(
    s=0.25,
):
    # Parameters
    a = 0.3
    b = 2.5

    # True function
    def y_true(x):
        return a + jnp.sin(x * 2 * jnp.pi / b)

    # Noisy function
    def y_noise(x, key):
        # return y_true(x) + s * jax.random.normal(key, x.shape)
        return y_true(x) + dist.Normal(0, s).sample(key, x.shape)

    return y_true, y_noise


# Simple flax MLP model
class MLP(nn.Module):
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        for hidden in self.layers:
            x = nn.Dense(hidden)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def make_bmlp_model(
    layer_size: list[int] = [5],
    prior: dist.Distribution | dict[str, dist.Distribution] = dist.Normal(0, 1.0),
    sigma_prior: dist.Distribution = dist.LogNormal(0, 0.1),
    use_bayesian: bool = False,
):
    # mlp = MLP(layer_size)

    def probabilistic_model_v2(x: jnp.ndarray, y: jnp.ndarray | None = None):
        # Expect x to be shape of (...batch_dims, 1)
        mlp = random_flax_module(
            "MLP",
            MLP(layer_size),
            input_shape=x.shape,
            prior=prior,
        )

        # Output shape is (..., 1)
        mu = mlp(x)
        # So we squeeze the last dimension
        mu = jnp.squeeze(mu, axis=-1)
        # sigma = numpyro.sample("sigma", dist.LogNormal(0, 0.1))
        sigma = numpyro.sample("sigma", sigma_prior)

        return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)  # type: ignore

    def probabilistic_model_v1(x: jnp.ndarray, y: jnp.ndarray | None = None):
        # Expect x to be shape of (...batch_dims, 1)

        mlp = flax_module(
            "MLP",
            MLP(layer_size),
            input_shape=x.shape,
        )

        # Output shape is (..., 1)
        mu = mlp(x)
        # So we squeeze the last dimension
        mu = jnp.squeeze(mu, axis=-1)
        sigma = numpyro.sample("sigma", dist.LogNormal(0, 0.1))

        return numpyro.sample("y", dist.Normal(mu, sigma), obs=y)  # type: ignore

    return probabilistic_model_v2 if use_bayesian else probabilistic_model_v1


def construct_normal_prior(
    posterior_samples: dict[str, jnp.ndarray],
) -> dict[str, dist.Distribution]:
    posterior_mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), posterior_samples)
    posterior_std = jax.tree.map(lambda x: jnp.std(x, axis=0), posterior_samples)

    prior = {}
    for name, mean in posterior_mean.items():
        prior[name] = dist.Normal(mean, posterior_std[name])

    return prior
