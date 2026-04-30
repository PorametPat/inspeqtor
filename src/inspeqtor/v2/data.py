import jax
import jax.numpy as jnp
from dataclasses import dataclass, field, asdict
import typing
from datetime import datetime
from pathlib import Path
import json
import polars as pl
import itertools


@jax.tree_util.register_dataclass
@dataclass
class DataBundled:
    control_params: jnp.ndarray
    unitaries: jnp.ndarray
    observables: jnp.ndarray
    aux: jnp.ndarray | None = None


@dataclass
class QubitInformation:
    """Dataclass to store qubit information

    Args:
        unit (str): The string representation of unit, currently support "GHz", "2piGHz", "2piHz", or "Hz".
        qubit_idx (int): the index of the qubit.
        anharmonicity (float): Anhamonicity of the qubit, kept for the sake of completeness.
        frequency (float): Qubit frequency.
        drive_strength (float): Drive strength of qubit, might be specific for IBMQ platform.

    Raises:
        ValueError: Fail to convert unit to GHz
    """

    unit: str
    qubit_idx: int
    anharmonicity: float
    frequency: float
    drive_strength: float
    date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __post_init__(self):
        self.convert_unit_to_ghz()

    def convert_unit_to_ghz(self):
        """Convert the unit of data stored in self to unit of GHz

        Raises:
            ValueError: Data stored in the unsupported unit
        """
        if self.unit == "GHz":
            pass
        elif self.unit == "Hz":
            self.anharmonicity = self.anharmonicity * 1e-9
            self.frequency = self.frequency * 1e-9
            self.drive_strength = self.drive_strength * 1e-9
        elif self.unit == "2piGHz":
            self.anharmonicity = self.anharmonicity / (2 * jnp.pi)
            self.frequency = self.frequency / (2 * jnp.pi)
            self.drive_strength = self.drive_strength / (2 * jnp.pi)
        elif self.unit == "2piHz":
            self.anharmonicity = self.anharmonicity / (2 * jnp.pi) * 1e-9
            self.frequency = self.frequency / (2 * jnp.pi) * 1e-9
            self.drive_strength = self.drive_strength / (2 * jnp.pi) * 1e-9
        else:
            raise ValueError("Unit must be GHz, 2piGHz, 2piHz, or Hz")

        # Set unit to GHz
        self.unit = "GHz"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict_qubit_info: dict):
        return cls(**dict_qubit_info)


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


def operator_map(operator: str) -> jnp.ndarray:
    match operator:
        case "X":
            return jnp.array([[0, 1], [1, 0]], dtype=complex)
        case "Y":
            return jnp.array([[0, -1j], [1j, 0]], dtype=complex)
        case "Z":
            return jnp.array([[1, 0], [0, -1]], dtype=complex)
        case "H":
            return jnp.array([[1, 1], [1, -1]], dtype=complex) / jnp.sqrt(2)
        case "S":
            return jnp.array([[1, 0], [0, 1j]], dtype=complex)
        case "Sdg":
            return jnp.array([[1, 0], [0, -1j]], dtype=complex)
        case "I":
            return jnp.array([[1, 0], [0, 1]], dtype=complex)
        case _:
            raise ValueError(
                f"Invalid operator label '{operator}'. Must be one of 'I', 'X', 'Y', 'Z', 'H', 'S', or 'Sdg'."
            )


def operator_from_label(ops: str) -> jnp.ndarray:
    return operator_map(ops)


def state_map(state: str) -> jnp.ndarray:
    match state:
        case "0":
            return jnp.array([1, 0], dtype=complex)
        case "1":
            return jnp.array([0, 1], dtype=complex)
        case "+":
            return jnp.array([1, 1], dtype=complex) / jnp.sqrt(2)
        case "-":
            return jnp.array([1, -1], dtype=complex) / jnp.sqrt(2)
        case "r":
            return jnp.array([1, 1j], dtype=complex) / jnp.sqrt(2)
        case "l":
            return jnp.array([1, -1j], dtype=complex) / jnp.sqrt(2)
        case _:
            raise ValueError(
                f"Invalid state label '{state}'. Must be one of '0', '1', '+', '-', 'r', or 'l'."
            )


def state_from_label(state: str, dm: bool) -> jnp.ndarray:
    vec = state_map(state).reshape(-1, 1)
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


def save_to_json(data: dict, path: typing.Union[str, Path]):
    """Save the dictionary as json to the path

    Args:
        data (dict): Dict to be save to file
        path (typing.Union[str, Path]): Path to save file.
    """
    if isinstance(path, str):
        path = Path(path)

    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


DataclassVar = typing.TypeVar("DataclassVar")


def read_from_json(
    path: typing.Union[str, Path],
    dataclass: typing.Union[None, type[DataclassVar]] = None,
) -> typing.Union[dict, DataclassVar]:
    """Construct provided `dataclass` instance with json file

    Args:
        path (typing.Union[str, Path]): Path to json file
        dataclass (typing.Union[None, type[DataclassVar]], optional): The constructor of the dataclass. Defaults to None.

    Returns:
        typing.Union[dict, DataclassVar]: Dataclass instance, if dataclass is not provideded, return dict instead.
    """
    if isinstance(path, str):
        path = Path(path)
    with open(path, "r") as f:
        config_dict = json.load(f)

    if dataclass is None:
        return config_dict
    else:
        return dataclass(**config_dict)
