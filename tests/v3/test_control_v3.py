import jax
import jax.numpy as jnp
from inspeqtor.experimental.ctyping import ParametersDictType
import typing


class Control(typing.NamedTuple):
    envelope: typing.Callable[[ParametersDictType, float | jnp.ndarray], jnp.ndarray]
    bound: dict[str, tuple]


def gaussian_control(
    A_max: float,
    drive_str: float,
    total_dt: int,
    dt: float,
    theta_range: tuple[float, float] = (0, 2 * jnp.pi),
):
    def envelope(param: ParametersDictType, t: float | jnp.ndarray) -> jnp.ndarray:
        sigma = 1 / (A_max * jnp.sqrt(2 * jnp.pi) * drive_str * dt)
        return (A_max * param["theta"] / (2 * jnp.pi)) * jnp.exp(
            -1 * ((t - (total_dt / 2)) ** 2 / (2 * (sigma) ** 2))
        )

    bound = {"theta": theta_range}

    return Control(envelope=envelope, bound=bound)


def sample(key: jnp.ndarray, bound: dict[str, tuple]):
    return {
        k: jax.random.uniform(
            jax.random.fold_in(key, idx), minval=value[0], maxval=value[1]
        )
        for idx, (k, value) in enumerate(bound.items())
    }


def ControlSequence(controls: dict[str, Control]):

    callables = {k: v.envelope for k, v in controls.items()}
    bounds = {k: v.bound for k, v in controls.items()}
    structure = []

    for ctrl in controls.keys():
        for ctrl_param in bounds[ctrl].keys():
            structure.append((ctrl, ctrl_param))

    def envelope(param: dict[str, ParametersDictType], t):
        return sum([ enve(param[k], t) for k, enve in callables.items() ])


    return envelope, bounds, structure


def test_control():
    control = gaussian_control(1.0, 0.1, 320, 0.1)

    key = jax.random.key(0)
    param = sample(key, control.bound)

    assert isinstance(param, dict)

    for k, v in param.items():
        assert isinstance(k, str) and isinstance(v, (float, jnp.ndarray))

    assert 0.0 < param['theta'] < 2 * jnp.pi
