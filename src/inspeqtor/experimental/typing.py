import typing
import jax.numpy as jnp
from flax.typing import FrozenVariableDict
from .decorator import deprecated

ParametersDictType = dict[str, typing.Union[float, jnp.ndarray]]
HamiltonianArgs = typing.TypeVar("HamiltonianArgs")

Wos = typing.Any | tuple[typing.Any, FrozenVariableDict | dict[str, typing.Any]]


@deprecated
def _ensure_wo_type(Wos: typing.Any) -> dict[str, jnp.ndarray]:
    if not isinstance(Wos, dict):
        raise TypeError(f"Expected Wos to be a dictionary, got {type(Wos)}")

    for key, value in Wos.items():
        if not isinstance(value, jnp.ndarray):
            raise TypeError(
                f"Expected the values of Wos to be jnp.ndarray, got {type(value)}"
            )

    return Wos


@deprecated
def ensure_wo_type(Wos: typing.Any) -> dict[str, jnp.ndarray]:
    assert isinstance(Wos, dict)
    assert all(
        [
            isinstance(key, str) and isinstance(value, jnp.ndarray)
            for key, value in Wos.items()
        ]
    )

    return Wos
