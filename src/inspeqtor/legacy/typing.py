import jax.numpy as jnp
from typing import Union, Protocol, Any
from flax.typing import VariableDict, FrozenVariableDict
import optax  # type: ignore
from dataclasses import dataclass
from enum import Enum


class LossChoice(Enum):
    MSEE = "MSEE"
    MAEF = "MAEF"


ParametersDictType = dict[str, float]

WoParams = Any | tuple[Any, FrozenVariableDict | dict[str, Any]]


class LossFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        training: bool,
    ) -> jnp.ndarray: ...


class TrainStepFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectations: jnp.ndarray,
        dropout_key: jnp.ndarray,
        transform_state: Union[None, optax.OptState],
    ) -> tuple[Any, Any, float]: ...


class TestStepFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        pulse_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectations: jnp.ndarray,
    ) -> float: ...


@dataclass
class HistoryEntry:
    epoch: int
    step: int
    batch_loss: float
    global_step: int
    epoch_loss: float | None = None
    test_loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None


@dataclass
class HistoryEntryV2(dict):
    MSEE: float | None
    MAEF: float | None
    step: int
    epoch: int
    loop: str


class CallbackFn(Protocol):
    def __call__(
        self,
        params: VariableDict,
        opt_state: optax.OptState,
        history: list[HistoryEntry],
    ) -> None: ...
