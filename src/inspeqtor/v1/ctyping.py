import typing
import jax.numpy as jnp
from flax.typing import FrozenVariableDict
import jax

ParametersDictType = dict[str, typing.Union[float, jnp.ndarray]]
HamiltonianArgs = typing.TypeVar("HamiltonianArgs")

Wos = typing.Any | tuple[typing.Any, FrozenVariableDict | dict[str, typing.Any]]

ArrayTree = typing.Union[
    jax.typing.ArrayLike,
    typing.Iterable["ArrayTree"],
    typing.Mapping[typing.Any, "ArrayTree"],
]
