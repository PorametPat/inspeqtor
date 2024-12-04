import typing
import jax.numpy as jnp
from flax.typing import FrozenVariableDict

ParametersDictType = dict[str, float]
HamiltonianArgs = typing.TypeVar("HamiltonianArgs")
WoParams = typing.Any | tuple[typing.Any, FrozenVariableDict | dict[str, typing.Any]]


def ensure_wo_params_type(Wos_params: typing.Any) -> dict[str, dict[str, jnp.ndarray]]:
    if not isinstance(Wos_params, dict):
        raise TypeError(
            f"Expected Wos_params to be a dictionary, got {type(Wos_params)}"
        )

    for key, value in Wos_params.items():
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected the values of Wos_params to be dictionaries, got {type(value)}"
            )

        for k, v in value.items():
            if not isinstance(v, jnp.ndarray):
                raise TypeError(
                    f"Expected the values of the values of Wos_params to be jnp.ndarray, got {type(v)}"
                )

    return Wos_params
