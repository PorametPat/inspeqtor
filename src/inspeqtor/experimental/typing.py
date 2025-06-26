import typing
import jax.numpy as jnp
from flax.typing import FrozenVariableDict


ParametersDictType = dict[str, typing.Union[float, jnp.ndarray]]
HamiltonianArgs = typing.TypeVar("HamiltonianArgs")

Wos = typing.Any | tuple[typing.Any, FrozenVariableDict | dict[str, typing.Any]]
