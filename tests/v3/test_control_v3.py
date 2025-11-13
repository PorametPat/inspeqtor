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


def _sample(key: jnp.ndarray, bound: dict[str, tuple]):
    return {
        k: jax.random.uniform(
            jax.random.fold_in(key, idx), minval=value[0], maxval=value[1]
        )
        for idx, (k, value) in enumerate(bound.items())
    }


def sample(
    key: jnp.ndarray, bounds: dict[str, dict[str, tuple]]
) -> dict[str, dict[str, jnp.ndarray]]:
    return {
        ctrl: _sample(jax.random.fold_in(key, idx), atomic_ctrl)
        for idx, (ctrl, atomic_ctrl) in enumerate(bounds.items())
    }


def ControlSequence(controls: dict[str, Control], structure: list | None = None):
    callables = {k: v.envelope for k, v in controls.items()}
    bounds = {k: v.bound for k, v in controls.items()}

    if structure is None:
        structure = []
        for ctrl in controls.keys():
            for ctrl_param in bounds[ctrl].keys():
                structure.append((ctrl, ctrl_param))

    def envelope(param: dict[str, ParametersDictType], t):
        return sum([enve(param[k], t) for k, enve in callables.items()])

    return envelope, bounds, structure


def test_control():
    control = gaussian_control(1.0, 0.1, 320, 0.1)

    key = jax.random.key(0)
    param = _sample(key, control.bound)

    assert isinstance(param, dict)

    for k, v in param.items():
        assert isinstance(k, str) and isinstance(v, (float, jnp.ndarray))

    assert 0.0 < param["theta"] < 2 * jnp.pi


def test_control_sequence():
    control = gaussian_control(1.0, 0.1, 320, 0.1)

    envelope, bounds, structure = ControlSequence({"g1": control, "g2": control})

    key = jax.random.key(0)
    param = sample(key, bounds)

    assert isinstance(param, dict)

    for k, v in param.items():
        assert isinstance(k, str) and isinstance(v, dict)

        for _k, _v in v.items():
            assert isinstance(_k, str) and isinstance(_v, (float, jnp.ndarray))

            # Check if the parameters is in the structure and within bound
            assert (k, _k) in structure

            assert bounds[k][_k][0] < _v < bounds[k][_k][1]
