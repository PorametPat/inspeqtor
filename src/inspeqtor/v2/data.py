import jax.numpy as jnp
from dataclasses import dataclass, field, asdict
import typing
from datetime import datetime
from pathlib import Path
import json
import polars as pl
import itertools

from ..v1.data import QubitInformation


@dataclass
class ExpectationValue:
    """Class representing a single experimental setting of state initialization and observable measurement.

    Supports both single-qubit and multi-qubit configurations using string representation:
    - Observable: "XYZ" (instead of ["X", "Y", "Z"])
    - Initial state: "+0r" (instead of ["+", "0", "r"])
    """

    initial_state: str
    # String where each character represents an observable for one qubit
    observable: str
    # String where each character represents an initial state for one qubit

    def __post_init__(self):
        # Ensure both strings have the same length (number of qubits)
        assert len(self.observable) == len(self.initial_state), (
            f"Observable and initial state must have same number of qubits: {len(self.observable)} != {len(self.initial_state)}"
        )

        # Validate observable characters
        for o in self.observable:
            assert o in "IXYZ", (
                f"Invalid observable '{o}'. Must be one of 'I', 'X', 'Y', or 'Z'"
            )

        # Validate initial state characters
        valid_states = "+-rl01"
        for s in self.initial_state:
            assert s in valid_states, (
                f"Invalid initial state '{s}'. Must be one of {valid_states}"
            )

    def to_dict(self):
        return {
            "initial_state": self.initial_state,
            "observable": self.observable,
        }

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExpectationValue):
            return False

        return (
            self.initial_state == __value.initial_state
            and self.observable == __value.observable
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __str__(self) -> str:
        return self.initial_state + "/" + self.observable


# Helper function for tensor products
def tensor_product(*operators) -> jnp.ndarray:
    """Create tensor product of multiple operators"""

    result = operators[0]
    for op in operators[1:]:
        result = jnp.kron(result, op)
    return result


operators_map = {
    "X": jnp.array([[0, 1], [1, 0]], dtype=jnp.complex_),
    "Y": jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex_),
    "Z": jnp.array([[1, 0], [0, -1]], dtype=jnp.complex_),
    "H": jnp.array([[1, 1], [1, -1]], dtype=jnp.complex_) / jnp.sqrt(2),
    "S": jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex_),
    "Sdg": jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex_),
    "I": jnp.array([[1, 0], [0, 1]], dtype=jnp.complex_),
}


def operator_from_label(ops: str) -> jnp.ndarray:
    return operators_map[ops]


state_map = {
    "0": jnp.array([1, 0], dtype=jnp.complex_),
    "1": jnp.array([0, 1], dtype=jnp.complex_),
    "+": jnp.array([1, 1], dtype=jnp.complex_) / jnp.sqrt(2),
    "-": jnp.array([1, -1], dtype=jnp.complex_) / jnp.sqrt(2),
    "r": jnp.array([1, 1j], dtype=jnp.complex_) / jnp.sqrt(2),
    "l": jnp.array([1, -1j], dtype=jnp.complex_) / jnp.sqrt(2),
}


def state_from_label(state: str, dm: bool) -> jnp.ndarray:
    vec = state_map[state].reshape(-1, 1)
    return vec if not dm else jnp.outer(vec, vec.conj())


def get_observable_operator(observable: str) -> jnp.ndarray:
    """Get the full observable operator as a tensor product"""
    ops = [operator_from_label(label) for label in observable]
    if len(ops) == 1:
        return ops[0]
    return tensor_product(*ops)


def get_initial_state(initial_state: str, dm: bool = True) -> jnp.ndarray:
    """Get the initial state as state vector or density matrix"""
    states = [state_from_label(label, dm=False) for label in initial_state]

    if len(states) == 1:
        state = states[0]
    else:
        # For multi-qubit state, compute the tensor product
        result = states[0]
        for s in states[1:]:
            result = jnp.kron(result, s)
        state = result

    # Convert to vector shape if needed
    if state.shape == (2, 1) or state.shape == (2 ** len(states), 1):
        # Already in correct shape
        pass
    elif state.shape == (2,) or state.shape == (2 ** len(states),):
        # Reshape to column vector
        state = state.reshape(-1, 1)

    if dm:
        return jnp.outer(state, state.conj())
    return state


def get_complete_expectation_values(
    num_qubits: int,
    observables: typing.Iterable[typing.Literal["I", "X", "Y", "Z"]] = [
        "I",
        "X",
        "Y",
        "Z",
    ],
    states: typing.Iterable[typing.Literal["+", "-", "r", "l", "0", "1"]] = [
        "+",
        "-",
        "r",
        "l",
        "0",
        "1",
    ],
    exclude_all_identities: bool = True,
) -> list[ExpectationValue]:
    """Generate a complete set of expectation values for characterizing a multi-qubit system"""

    # For n qubits, we need all combinations of observables and states
    result: typing.Iterable[ExpectationValue] = []

    # Generate all combinations of observables
    for obs_combo in itertools.product(observables, repeat=num_qubits):
        for state_combo in itertools.product(states, repeat=num_qubits):
            obs_str = "".join(obs_combo)
            state_str = "".join(state_combo)
            result.append(ExpectationValue(observable=obs_str, initial_state=state_str))

    if exclude_all_identities:
        result = [exp for exp in result if exp.observable != "I" * num_qubits]

    return result


@dataclass
class ExperimentConfiguration:
    """Experiment configuration dataclass"""

    qubits: typing.Sequence[QubitInformation]
    expectation_values_order: typing.Sequence[ExpectationValue]
    parameter_structure: typing.Sequence[
        typing.Sequence[str]
    ]  # Get from the pulse sequence .get_parameter_names()
    backend_name: str
    shots: int
    EXPERIMENT_IDENTIFIER: str
    EXPERIMENT_TAGS: typing.Sequence[str]
    description: str
    device_cycle_time_ns: float
    sequence_duration_dt: int
    sample_size: int
    date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    additional_info: dict[str, str | int | float] = field(default_factory=dict)

    def to_dict(self):
        return {
            **asdict(self),
            "qubits": [qubit.to_dict() for qubit in self.qubits],
            "expectation_values_order": [
                exp.to_dict() for exp in self.expectation_values_order
            ],
        }

    @classmethod
    def from_dict(cls, dict_experiment_config):
        dict_experiment_config["qubits"] = [
            QubitInformation.from_dict(qubit)
            for qubit in dict_experiment_config["qubits"]
        ]

        dict_experiment_config["expectation_values_order"] = [
            ExpectationValue.from_dict(exp)
            for exp in dict_experiment_config["expectation_values_order"]
        ]

        dict_experiment_config["parameter_structure"] = [
            tuple(control) for control in dict_experiment_config["parameter_structure"]
        ]

        return cls(**dict_experiment_config)

    def to_file(self, path: typing.Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)

        # os.makedirs(path, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(cls, path: typing.Union[Path, str]):
        if isinstance(path, str):
            path = Path(path)
        with open(path / "config.json", "r") as f:
            dict_experiment_config = json.load(f)

        return cls.from_dict(dict_experiment_config)

    def __str__(self):
        lines = [
            "=" * 60,
            "EXPERIMENT CONFIGURATION",
            "=" * 60,
            f"Identifier: {self.EXPERIMENT_IDENTIFIER}",
            f"Backend: {self.backend_name}",
            f"Date: {self.date}",
            f"Description: {self.description}",
            "",
            f"Shots: {self.shots:,}",
            f"Sample Size: {self.sample_size}",
            f"Device Cycle Time: {self.device_cycle_time_ns:.4f} ns",
            f"Sequence Duration: {self.sequence_duration_dt} dt",
            "",
            f"Qubits: {len(self.qubits)}",
            *[f"  - {qubit}" for qubit in self.qubits],
            "",
            f"Expectation Values: {len(self.expectation_values_order)}",
            f"  (States: {set(e.initial_state for e in self.expectation_values_order)})",
            f"  (Observables: {set(e.observable for e in self.expectation_values_order)})",
            "",
            f"Parameter Structure: {self.parameter_structure}",
            f"Tags: {', '.join(self.EXPERIMENT_TAGS)}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class ExperimentalData:
    """Dataclass for processing of the characterization dataset.
    A difference between preprocess and postprocess dataset is that postprocess group
    expectation values same control parameter id within single row instead of multiple rows.
    """

    config: ExperimentConfiguration
    parameter_dataframe: pl.DataFrame
    observed_dataframe: pl.DataFrame
    mode: typing.Literal["expectation_value", "binary"] = "expectation_value"

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert "parameter_id" in self.parameter_dataframe
        assert "parameter_id" in self.observed_dataframe

        assert (
            self.parameter_dataframe["parameter_id"]
            .unique()
            .sort()
            .equals(self.observed_dataframe["parameter_id"].unique().sort())
        )

    def get_parameter(self) -> jnp.ndarray:
        col_selector = ["/".join(param) for param in self.config.parameter_structure]
        return self.parameter_dataframe[col_selector].to_jax("array")

    def get_observed(self) -> jnp.ndarray:
        col_selector = [str(expval) for expval in self.config.expectation_values_order]

        if self.mode == "binary":
            return jnp.array(
                [
                    calculate_expectation_value_from_binary_dataframe(
                        str(exp), self.observed_dataframe
                    )
                    for exp in self.config.expectation_values_order
                ]
            ).transpose()

        return self.observed_dataframe[col_selector].to_jax("array")

    def save_to_folder(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        self.config.to_file(path)

        self.parameter_dataframe.write_csv(path / "parameter.csv")
        self.observed_dataframe.write_csv(path / "observed.csv")

    @classmethod
    def from_folder(cls, path: str | Path) -> "ExperimentalData":
        if isinstance(path, str):
            path = Path(path)

        config = ExperimentConfiguration.from_file(path)
        parameter_dataframe = pl.read_csv(path / "parameter.csv")
        observed_dataframe = pl.read_csv(path / "observed.csv")

        return cls(
            config=config,
            parameter_dataframe=parameter_dataframe,
            observed_dataframe=observed_dataframe,
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExperimentalData):
            return False

        return (
            self.config == __value.config
            and self.parameter_dataframe.equals(__value.parameter_dataframe)
            and self.observed_dataframe.equals(__value.observed_dataframe)
        )

    def __str__(self):
        lines = [
            "=" * 60,
            "EXPERIMENTAL DATA",
            str(self.config),
            "",
            "Parameter DataFrame",
            str(self.parameter_dataframe),
            "",
            "Observed DataFrame",
            str(self.observed_dataframe),
            "=" * 60,
        ]
        return "\n".join(lines)


# def check_parity(n: int):
#     """
#     Determines the parity of a number.

#     Args:
#         n (int): The input integer.

#     Returns:
#         int: 0 if the number has even parity, 1 if it has odd parity.
#     """
#     parity = 0
#     while n != 0:
#         parity ^= n & 1  # XOR the current LSB with parity
#         n >>= 1  # Right shift to process the next bit
#     return parity


def check_parity(n):
    """
    Determines the parity of a number using bitwise_count.

    Efficiently computes parity by counting all 1 bits and taking modulo 2.
    This is much faster than the iterative approach as it uses hardware
    intrinsics for population count.

    Args:
        n: The input integer.

    Returns:
        0 if the number has even parity, 1 if it has odd parity.

    Example:
        >>> check_parity(7)  # 0b111 -> three 1s -> odd parity
        1
        >>> check_parity(6)  # 0b110 -> two 1s -> even parity
        0
    """
    return jnp.bitwise_count(n) % 2


def calculate_expectation_value_from_binary_dataframe(
    expvals: str, dataframe: pl.DataFrame
) -> jnp.ndarray:
    matching_cols = [col for col in dataframe.columns if col.startswith(expvals)]

    # +1 eigenvalue
    even_parity = (
        dataframe.select(
            [col for col in matching_cols if check_parity(int(col.split("/")[-1])) == 0]
        )
        .to_jax("array")
        .sum(-1)
    )

    # -1 eigenvalue
    odd_parity = (
        dataframe.select(
            [col for col in matching_cols if check_parity(int(col.split("/")[-1])) == 1]
        )
        .to_jax("array")
        .sum(-1)
    )

    expectation_value = (even_parity - odd_parity) / (even_parity + odd_parity)

    return expectation_value
