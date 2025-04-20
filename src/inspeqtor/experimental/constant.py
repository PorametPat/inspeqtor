from .data import Operator, ExpectationValue, State
import jax.numpy as jnp

X = Operator.from_label("X")
Y = Operator.from_label("Y")
Z = Operator.from_label("Z")


plus_projectors = {
    "X": State.from_label("+", dm=True),
    "Y": State.from_label("r", dm=True),
    "Z": State.from_label("0", dm=True),
}

default_expectation_values_order = [
    ExpectationValue(observable="X", initial_state="+"),
    ExpectationValue(observable="X", initial_state="-"),
    ExpectationValue(observable="X", initial_state="r"),
    ExpectationValue(observable="X", initial_state="l"),
    ExpectationValue(observable="X", initial_state="0"),
    ExpectationValue(observable="X", initial_state="1"),
    ExpectationValue(observable="Y", initial_state="+"),
    ExpectationValue(observable="Y", initial_state="-"),
    ExpectationValue(observable="Y", initial_state="r"),
    ExpectationValue(observable="Y", initial_state="l"),
    ExpectationValue(observable="Y", initial_state="0"),
    ExpectationValue(observable="Y", initial_state="1"),
    ExpectationValue(observable="Z", initial_state="+"),
    ExpectationValue(observable="Z", initial_state="-"),
    ExpectationValue(observable="Z", initial_state="r"),
    ExpectationValue(observable="Z", initial_state="l"),
    ExpectationValue(observable="Z", initial_state="0"),
    ExpectationValue(observable="Z", initial_state="1"),
]

SX = (jnp.exp(1j * jnp.pi / 4) / jnp.sqrt(2)) * jnp.array([[1, -1j], [-1j, 1]])
