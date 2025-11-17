from .data import ExpectationValue, get_observable_operator, get_initial_state
import jax.numpy as jnp

X = get_observable_operator("X")
Y = get_observable_operator("Y")
Z = get_observable_operator("Z")


plus_projectors = {
    "X": get_initial_state("+", dm=True),
    "Y": get_initial_state("r", dm=True),
    "Z": get_initial_state("0", dm=True),
}

minus_projectors = {
    "X": get_initial_state("-", dm=True),
    "Y": get_initial_state("l", dm=True),
    "Z": get_initial_state("1", dm=True),
}


def get_default_expectation_values_order():
    return [
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


default_expectation_values_order = get_default_expectation_values_order()


SX = (jnp.exp(1j * jnp.pi / 4) / jnp.sqrt(2)) * jnp.array([[1, -1j], [-1j, 1]])
