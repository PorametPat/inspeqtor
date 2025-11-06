import jax.numpy as jnp
from dataclasses import dataclass, field, asdict
import typing
from datetime import datetime
from pathlib import Path
import json
import polars as pl
from functools import cached_property
import itertools

from ..experimental.data import QubitInformation, Operator, State


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
    expectation_value: float | None = None

    def __post_init__(self):
        # Ensure both strings have the same length (number of qubits)
        assert (
            len(self.observable) == len(self.initial_state)
        ), f"Observable and initial state must have same number of qubits: {len(self.observable)} != {len(self.initial_state)}"

        # Validate observable characters
        for o in self.observable:
            assert (
                o in "XYZ"
            ), f"Invalid observable '{o}'. Must be one of 'X', 'Y', or 'Z'"

        # Validate initial state characters
        valid_states = "+-rl01"
        for s in self.initial_state:
            assert (
                s in valid_states
            ), f"Invalid initial state '{s}'. Must be one of {valid_states}"

    @cached_property
    def num_qubits(self) -> int:
        """Return the number of qubits in this experimental setting"""
        return len(self.observable)

    @cached_property
    def observable_list(self) -> list[str]:
        """Get the observable as a list of strings, one per qubit"""
        return [o for o in self.observable]

    @cached_property
    def initial_state_list(self) -> list[str]:
        """Get the initial state as a list of strings, one per qubit"""
        return [s for s in self.initial_state]

    @cached_property
    def initial_statevector(self) -> jnp.ndarray:
        return self.get_initial_state(dm=False)

    @cached_property
    def initial_density_matrix(self) -> jnp.ndarray:
        return self.get_initial_state(dm=True)

    @cached_property
    def observable_matrix(self) -> jnp.ndarray:
        return self.get_observable_operator()

    def get_observable_operator(self) -> jnp.ndarray:
        """Get the full observable operator as a tensor product"""
        ops = [Operator.from_label(label) for label in self.observable_list]
        if len(ops) == 1:
            return ops[0]
        return tensor_product(*ops)

    def get_initial_state(self, dm: bool = True) -> jnp.ndarray:
        """Get the initial state as state vector or density matrix"""
        states = [
            State.from_label(label, dm=False) for label in self.initial_state_list
        ]

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

    def to_dict(self):
        return {
            "initial_state": self.initial_state,
            "observable": self.observable,
            "expectation_value": self.expectation_value,
        }

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExpectationValue):
            return False

        return (
            self.initial_state == __value.initial_state
            and self.observable == __value.observable
            and self.expectation_value == __value.expectation_value
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def __str__(self) -> str:
        return self.initial_state + "/" + self.observable


# Helper function for tensor products
def tensor_product(*operators) -> jnp.ndarray:
    """Create tensor product of multiple operators"""
    if not operators:
        raise ValueError("Need at least one operator")

    result = operators[0]
    for op in operators[1:]:
        result = jnp.kron(result, op)
    return result


def get_complete_expectation_values(num_qubits: int) -> list[ExpectationValue]:
    """Generate a complete set of expectation values for characterizing a multi-qubit system"""
    observables = ["X", "Y", "Z"]
    states = ["+", "-", "r", "l", "0", "1"]

    # For n qubits, we need all combinations of observables and states
    result = []

    # Generate all combinations of observables
    for obs_combo in itertools.product(observables, repeat=num_qubits):
        for state_combo in itertools.product(states, repeat=num_qubits):
            obs_str = "".join(obs_combo)
            state_str = "".join(state_combo)
            result.append(ExpectationValue(observable=obs_str, initial_state=state_str))

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
    instance: str
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
            f"Backend: {self.backend_name} (instance: {self.instance})",
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
        col_selector = [
            f"{expval.initial_state}/{expval.observable}"
            for expval in self.config.expectation_values_order
        ]
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
