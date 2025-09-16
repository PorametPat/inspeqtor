from dataclasses import dataclass, asdict, field
from datetime import datetime
import jax
import jax.numpy as jnp
import typing
import json
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from numpyro.contrib.module import ParamShape

from .ctyping import ParametersDictType


def add_hilbert_level(op: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Add a level to the operator or state

    Args:
        op (jnp.ndarray): The qubit operator or state
        is_state (bool): True if the operator is a state, False if the operator is an operator

    Returns:
        jnp.ndarray: The qutrit operator or state
    """
    return jax.scipy.linalg.block_diag(op, x)


@dataclass
class Operator:
    """Dataclass for accessing qubit operators. Support X, Y, Z, Hadamard, S, Sdg, and I gate.

    Raises:
        ValueError: Provided operator is not supperted

    """

    _pauli_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    _pauli_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    _pauli_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    _hadamard = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)
    _s_gate = jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)
    _sdg_gate = jnp.array([[1, 0], [0, -1j]], dtype=jnp.complex64)
    _identity = jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)

    @classmethod
    def from_label(cls, op: str) -> jnp.ndarray:
        """Initialize the operator from the label

        Args:
            op (str): The label of the operator

        Raises:
            ValueError: Operator not supported

        Returns:
            jnp.ndarray: The operator
        """

        if op == "X":
            operator = cls._pauli_x
        elif op == "Y":
            operator = cls._pauli_y
        elif op == "Z":
            operator = cls._pauli_z
        elif op == "H":
            operator = cls._hadamard
        elif op == "S":
            operator = cls._s_gate
        elif op == "Sdg":
            operator = cls._sdg_gate
        elif op == "I":
            operator = cls._identity
        else:
            raise ValueError(f"Operator {op} is not supported")

        return operator

    @classmethod
    def to_qutrit(cls, op: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
        """Add extra dimension to the operator

        Args:
            op (jnp.ndarray): Qubit operator
            value (float, optional): Value to be add at the extra dimension diagonal entry. Defaults to 1.0.

        Returns:
            jnp.ndarray: New operator for qutrit space.
        """
        return add_hilbert_level(op, x=jnp.array([value]))


@dataclass
class State:
    """Dataclass for accessing eigenvector corresponded to eigenvalue of Pauli operator X, Y, and Z.

    Raises:
        ValueError: Provided state is not supported
        ValueError: Provided state is not qubit
    """

    _zero = jnp.array([1, 0], dtype=jnp.complex64)
    _one = jnp.array([0, 1], dtype=jnp.complex64)
    _plus = jnp.array([1, 1], dtype=jnp.complex64) / jnp.sqrt(2)
    _minus = jnp.array([1, -1], dtype=jnp.complex64) / jnp.sqrt(2)
    _right = jnp.array([1, 1j], dtype=jnp.complex64) / jnp.sqrt(2)
    _left = jnp.array([1, -1j], dtype=jnp.complex64) / jnp.sqrt(2)

    @classmethod
    def from_label(cls, state: str, dm: bool = False) -> jnp.ndarray:
        """Initialize the state from the label

        Args:
            state (str): The label of the state
            dm (bool, optional): Initialized as statevector or density matrix. Defaults to False.

        Raises:
            ValueError: State not supported

        Returns:
            jnp.ndarray: The state
        """

        if state in ["0", "Z+"]:
            state_vec = cls._zero
        elif state in ["1", "Z-"]:
            state_vec = cls._one
        elif state in ["+", "X+"]:
            state_vec = cls._plus
        elif state in ["-", "X-"]:
            state_vec = cls._minus
        elif state in ["r", "Y+"]:
            state_vec = cls._right
        elif state in ["l", "Y-"]:
            state_vec = cls._left
        else:
            raise ValueError(f"State {state} is not supported")

        state_vec = state_vec.reshape(2, 1)

        return state_vec if not dm else jnp.outer(state_vec, state_vec.conj())

    @classmethod
    def to_qutrit(cls, state: jnp.ndarray) -> jnp.ndarray:
        """Promote qubit state to qutrit with zero probability

        Args:
            state (jnp.ndarray): Density matrix of 2 x 2 qubit state.

        Raises:
            ValueError: Provided state is not qubit

        Returns:
            jnp.ndarray: Qutrit density matrix
        """
        if state.shape != (2, 2):
            raise ValueError("Shape of the state is not as expected, expect (2, 2)")

        return add_hilbert_level(state, x=jnp.array([0.0]))


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
    """Dataclass to store expectation value information

    Args:
        initial_state (str): String representation of inital state. Currently support "+", "-", "r", "l", "0", "1".
        observable (str): String representation of quantum observable.  Currently support "X", "Y", "Z".
        expectation_value (None | float): the expectation value. Default to None

    Raises:
        ValueError: Not support initial state
        ValueError: Not support observable
        ValueError: Not support initial state
        ValueError: Not support observable
    """

    initial_state: str
    observable: str
    expectation_value: None | float = None

    # Not serialized
    initial_statevector: jnp.ndarray = field(init=False)
    initial_density_matrix: jnp.ndarray = field(init=False)
    observable_matrix: jnp.ndarray = field(init=False)

    def __post_init__(self):
        if self.initial_state not in ["+", "-", "r", "l", "0", "1"]:
            raise ValueError(f"Initial state {self.initial_state} is not supported")
        if self.observable not in ["X", "Y", "Z"]:
            raise ValueError(f"Observable {self.observable} is not supported")

        self.initial_statevector = State.from_label(self.initial_state)
        self.initial_density_matrix = State.from_label(self.initial_state, dm=True)
        self.observable_matrix = Operator.from_label(self.observable)

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

    def __str__(self):
        return f"{self.initial_state}/{self.observable} = {self.expectation_value}"

    # Overwrite the __repr__ method of the class
    def __repr__(self):
        return f'{self.__class__.__name__}(initial_state="{self.initial_state}", observable="{self.observable}", expectation_value={self.expectation_value})'

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class ExperimentConfiguration:
    """Experiment configuration dataclass"""

    qubits: typing.Sequence[QubitInformation]
    expectation_values_order: typing.Sequence[ExpectationValue]
    parameter_names: typing.Sequence[
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


@dataclass
class PredefinedCol:
    name: str
    description: str
    type: type
    unique: bool = False
    required: bool = True
    checks: list[typing.Callable[[typing.Union[str, float, int]], bool]] = field(
        default_factory=list
    )


EXPECTATION_VALUE = PredefinedCol(
    name="expectation_value",
    description="The value of the expectation value, must be set later after the experiment.",
    type=float,
    checks=[lambda x: x >= -1 and x <= 1 if isinstance(x, (int, float)) else False],
)

INITIAL_STATE = PredefinedCol(
    name="initial_state",
    description="The initial state (+, -, r, l, 0, 1) of the circuit.",
    type=str,
    checks=[lambda x: x in ["+", "-", "r", "l", "0", "1"]],
)

OBSERVABLE = PredefinedCol(
    name="observable",
    description="The Pauli operator (X, Y, Z) that the circuit measure expectation value on.",
    type=str,
    checks=[lambda x: x in ["X", "Y", "Z"]],
)

PARAMETERS_ID = PredefinedCol(
    name="parameters_id",
    description="The id of the parameters, must be unique for each row of the postprocessed_data.",
    type=int,
)

REQUIRED_COLUMNS = [
    EXPECTATION_VALUE,
    INITIAL_STATE,
    OBSERVABLE,
    PARAMETERS_ID,
]


def make_row(
    expectation_value: float,
    initial_state: str,
    observable: str,
    parameters_id: int,
    parameters_list: list[ParametersDictType],
    **kwargs,
) -> dict[str, typing.Union[float, int, str]]:
    # Prefix the parameters_list keys with f"parameter/{i}/" for i in range(len(parameters_list))
    # and flatten the dict into a single dict
    parameters_dict = {}
    for i, params in enumerate(parameters_list):
        for k, v in params.items():
            parameters_dict[f"parameter/{i}/{k}"] = float(v)

    # Validate that kwargs does not have the same key as parameters_dict and serializable
    assert all([k not in parameters_dict.keys() for k in kwargs.keys()])
    assert all([isinstance(k, str) for k in kwargs.keys()])
    assert all([isinstance(v, (int, float, str)) for v in kwargs.values()])

    return {
        # Require for postprocessing
        EXPECTATION_VALUE.name: expectation_value,
        INITIAL_STATE.name: initial_state,
        OBSERVABLE.name: observable,
        PARAMETERS_ID.name: parameters_id,
        **parameters_dict,
        **kwargs,
    }


def flatten_parameter_name_with_prefix(
    parameter_names: typing.Sequence[typing.Sequence[str]],
) -> list[str]:
    """Create a flatten list of parameter names with prefix parameter/{i}/

    Args:
        parameter_names (typing.Sequence[typing.Sequence[str]]): The list of parameter names from the pulse sequence
                                                                 or the experiment configuration

    Returns:
        list[str]: The flatten list of parameter names with prefix parameter/{i}/
    """
    return [
        f"parameter/{i}/{name}"
        for i, names in enumerate(parameter_names)
        for name in names
    ]


def transform_parameter_name(name: str) -> str:
    """Remove "parameter/{i}/" from provided name

    Args:
        name (str): Name of the control parameters

    Returns:
        str: Name that have "parameter/{i}/" strip.
    """
    if name.startswith("parameter/"):
        return "/".join(name.split("/")[2:])
    else:
        return name


def get_parameters_dict_list(
    parameters_name: typing.Sequence[typing.Sequence[str]], parameters_row: pd.Series
) -> list[ParametersDictType]:
    """Get the list of dict containing name and value of each control in the sequence.

    Args:
        parameters_name (typing.Sequence[typing.Sequence[str]]): _description_
        parameters_row (pd.Series): _description_

    Returns:
        list[ParametersDictType]: _description_
    """
    recovered_parameters: list[ParametersDictType] = [
        {
            # Split to remove the parameter/{i}/
            transform_parameter_name(k): v
            for k, v in parameters_row.items()
            # Check if the key is parameter/{i}/ and the value is float
            if isinstance(k, str)
            and k.startswith(f"parameter/{i}/")
            and isinstance(v, (float, int))
        }
        for i in range(len(parameters_name))
    ]

    return recovered_parameters


@dataclass
class ExperimentData:
    """Dataclass for processing of the characterization dataset.
    A difference between preprocess and postprocess dataset is that postprocess group
    expectation values same control parameter id within single row instead of multiple rows.

    Args:
        experiment_config (ExperimentConfiguration): Experiment configuration
        preprocess_data (pd.DataFrame): Pandas dataframe containing the preprocess dataset
        _postprocessed_data: (pd.DataFrame): Provide this optional argument to skip dataset postprocessing.
        keep_decimal (int): the precision of floating point to keep.
    """

    experiment_config: ExperimentConfiguration
    preprocess_data: pd.DataFrame
    # optional
    _postprocessed_data: pd.DataFrame | None = field(default=None)

    # Setting
    keep_decimal: int = 10
    # Postprocessing
    postprocessed_data: pd.DataFrame = field(init=False)
    parameter_columns: list[str] = field(init=False)
    parameters: np.ndarray = field(init=False)

    def __post_init__(self):
        self.preprocess_data = self.preprocess_data.round(self.keep_decimal)

        # Validate that self.preprocess_data have all the required columns
        self.validate_preprocess_data()
        logging.info("Preprocess data validated")

        if self._postprocessed_data is not None:
            self.postprocessed_data = self._postprocessed_data
            logging.info("Postprocess data set")

        else:
            post_data = self.transform_preprocess_data_to_postprocess_data()
            logging.info("Preprocess data transformed to postprocess data")

            self.postprocessed_data = post_data.round(self.keep_decimal)

        # Validate the data with schema
        self.validate_postprocess_data(self.postprocessed_data)
        logging.info("Postprocess data validated")

        self.parameter_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )
        num_features = len(self.experiment_config.parameter_names[0])
        num_controls = len(self.experiment_config.parameter_names)

        try:
            temp_params = np.array(
                self.postprocessed_data[self.parameter_columns]
                .to_numpy()
                .reshape(
                    (self.experiment_config.sample_size, num_controls, num_features)
                )
            )
        except Exception:
            logging.info(
                "Could not reshape parameters with shape (sample_size, num_controls, num_features), automatically reshaping to (sample_size, -1)"
            )
            temp_params = np.array(
                self.postprocessed_data[self.parameter_columns]
                .to_numpy()
                .reshape((self.experiment_config.sample_size, -1))
            )
            logging.info(f"Parameters reshaped to {temp_params.shape}")

        self.parameters = temp_params

        logging.info("Parameters converted to numpy array")

        assert (
            self.preprocess_data[self.parameter_columns]
            .drop_duplicates(ignore_index=True)
            .equals(
                self.postprocessed_data[self.parameter_columns].drop_duplicates(
                    ignore_index=True
                )
            )
        ), "The preprocess_data and postprocessed_data does not have the same parameters."
        logging.info("Preprocess data and postprocess data have the same parameters")

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ExperimentData):
            return False

        return (
            self.experiment_config == __value.experiment_config
            and self.preprocess_data.equals(__value.preprocess_data)
        )

    def validate_preprocess_data(self):
        """Validate that the preprocess_data have all the required columns.

        Required columns:
            - EXPECTATION_VALUE
            - INITIAL_STATE
            - OBSERVABLE
            - PARAMETERS_ID
        """
        for col in REQUIRED_COLUMNS:
            if col.required:
                assert (
                    col.name in self.preprocess_data.columns
                ), f"Column {col.name} is required but not found in the preprocess_data."

        # Validate that the preprocess_data have all expected parameters columns
        required_parameters_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )

        for _col in required_parameters_columns:
            assert (
                _col in self.preprocess_data.columns
            ), f"Column {_col} is required but not found in the preprocess_data."

    def validate_postprocess_data(self, post_data: pd.DataFrame):
        """Validate postprocess dataset, by check the requirements given by `PredefinedCol` instance of each column
        that required in the postprocessed dataset.

        Args:
            post_data (pd.DataFrame): Postprocessed dataset to be validated.
        """
        logging.info("Validating postprocess data")
        # Validate that the postprocess_data have all the required columns
        for col in REQUIRED_COLUMNS:
            if col.required:
                assert (
                    col.name in post_data.columns
                ), f"Column {col.name} is required but not found in the postprocess_data."

        # Validate the check functions
        for col in REQUIRED_COLUMNS:
            for check in col.checks:
                assert all(
                    [check(v) for v in post_data[col.name]]
                ), f"Column {col.name} failed the check function {check}"

        # Validate that the postprocess_data have all expected parameters columns
        required_parameters_columns = flatten_parameter_name_with_prefix(
            self.experiment_config.parameter_names
        )
        for _col in required_parameters_columns:
            assert (
                _col in post_data.columns
            ), f"Column {_col} is required but not found in the postprocess_data."

    def transform_preprocess_data_to_postprocess_data(self) -> pd.DataFrame:
        """Internal method to post process the dataset.

        Todo:
            Use new experimental implementation from_long to wide dataframe

        Raises:
            ValueError: There is duplicate entry of the expectation value.

        Returns:
            pd.DataFrame: Postprocessed experiment dataset.
        """
        # Postprocess the data squeezing the data into the expectation values
        # Required columns: PARAMETERS_ID, OBSERVABLE, INITIAL_STATE, EXPECTATION_VALUE, + experiment_config.parameter_names
        post_data = []

        for params_id in range(self.experiment_config.sample_size):
            # NOTE: Assume that parameters_id starts from 0 and is continuous to sample_size - 1
            rows = self.preprocess_data.loc[
                self.preprocess_data[PARAMETERS_ID.name] == params_id
            ]

            expectation_values = {}
            for _, exp_order in enumerate(
                self.experiment_config.expectation_values_order
            ):
                expectation_value = rows.loc[
                    (rows[OBSERVABLE.name] == exp_order.observable)
                    & (rows[INITIAL_STATE.name] == exp_order.initial_state)
                ][EXPECTATION_VALUE.name].values

                if expectation_value.shape[0] != 1:
                    raise ValueError(
                        f"Expectation value for params_id {params_id}, initial_state {exp_order.initial_state}, observable {exp_order.observable} is not unique. The length is {len(expectation_value)}."
                    )

                expectation_values[
                    f"{EXPECTATION_VALUE.name}/{exp_order.initial_state}/{exp_order.observable}"
                ] = expectation_value[0]

            drop_duplicates_row = rows.drop_duplicates(
                subset=flatten_parameter_name_with_prefix(
                    self.experiment_config.parameter_names
                )
            )
            # Assert that only one row is returned
            assert drop_duplicates_row.shape[0] == 1
            pulse_parameters = drop_duplicates_row.to_dict(orient="records")[0]

            new_row = {
                PARAMETERS_ID.name: params_id,
                **expectation_values,
                **{str(k): v for k, v in pulse_parameters.items()},
            }

            post_data.append(new_row)

        return pd.DataFrame(post_data)

    def get_parameters_dataframe(self) -> pd.DataFrame:
        """Get dataframe with only the columns of control parameters.

        Returns:
            pd.DataFrame: Dataframe with only the columns of control parameters.
        """
        return self.postprocessed_data[self.parameter_columns]

    def get_expectation_values(self) -> np.ndarray:
        """Get the expectation value of the shape (sample_size, num_expectation_value)

        Returns:
            np.ndarray: expectation value of the shape (sample_size, num_expectation_value)
        """
        expectation_value = self.postprocessed_data[
            [
                f"expectation_value/{col.initial_state}/{col.observable}"
                for col in self.experiment_config.expectation_values_order
            ]
        ].to_numpy()

        return np.array(expectation_value)

    def get_parameters_dict_list(self) -> list[list[ParametersDictType]]:
        """Get the list, where each element is list of dict of the control parameters of the dataset.

        Returns:
            list[list[ParametersDictType]]: The list of list of dict of parameter.
        """
        _temp = self.postprocessed_data[self.parameter_columns]

        _params_list = [
            get_parameters_dict_list(self.experiment_config.parameter_names, row)
            for _, row in _temp.iterrows()
        ]

        return _params_list

    def save_to_folder(self, path: typing.Union[Path, str]):
        """Save the experiment data to given folder

        Args:
            path (typing.Union[Path, str]): Path of the folder for experiment data to be saved.
        """
        if isinstance(path, str):
            path = Path(path)

        # os.makedirs(path, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        self.experiment_config.to_file(path)
        self.preprocess_data.to_csv(path / "preprocess_data.csv", index=False)
        self.postprocessed_data.to_csv(path / "postprocessed_data.csv", index=False)

    @classmethod
    def from_folder(cls, path: typing.Union[Path, str]) -> "ExperimentData":
        """Read the experiment data from path

        Args:
            path (typing.Union[Path, str]): path to the folder contain experiment data. Expected to be used with `self.save_to_folder` method.

        Returns:
            ExperimentData: Intance of `ExperimentData` read from path.
        """
        if isinstance(path, str):
            path = Path(path)

        experiment_config = ExperimentConfiguration.from_file(path)
        preprocess_data = pd.read_csv(
            path / "preprocess_data.csv",
        )

        # Check if postprocessed_data exists
        if not (path / "postprocessed_data.csv").exists():
            # if not os.path.exists(path / "postprocessed_data.csv"):
            postprocessed_data = None
        else:
            postprocessed_data = pd.read_csv(
                path / "postprocessed_data.csv",
            )

        return cls(
            experiment_config=experiment_config,
            preprocess_data=preprocess_data,
            _postprocessed_data=postprocessed_data,
        )

    def analysis_sum_of_expectation_values(self) -> pd.DataFrame:
        paulis = ["X", "Y", "Z"]
        initial_states = [("0", "1"), ("+", "-"), ("r", "l")]
        data = {}
        for pauli in paulis:
            for initial_state in initial_states:
                _name = f"{pauli}/{initial_state[0]}/{initial_state[1]}"

                res = (
                    self.postprocessed_data[
                        f"expectation_value/{initial_state[0]}/{pauli}"
                    ]
                    + self.postprocessed_data[
                        f"expectation_value/{initial_state[1]}/{pauli}"
                    ]
                )

                data[_name] = res.to_numpy()

        return pd.DataFrame(data)


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


def recursive_parse(data: dict, parse_fn):
    temp_dict = {}
    for key, value in data.items():
        is_leaf, parse_value = parse_fn(key, value)

        if not is_leaf:
            parse_value = recursive_parse(value, parse_fn)

        temp_dict[key] = parse_value

    return temp_dict


def default_parse_fn(key, value):
    if isinstance(value, list):
        return True, jnp.array(value)
    elif isinstance(value, dict) and len(value) == 1 and "shape" in value:
        return True, ParamShape(shape=tuple(value["shape"]))
    elif isinstance(value, dict):
        return False, None
    else:
        return True, value


def load_pytree_from_json(path: str | Path, parse_fn=default_parse_fn):
    """Load pytree from json

    Args:
        path (str | Path): Path to JSON file containing pytree
        array_keys (list[str], optional): list of key to convert to jnp.numpy. Defaults to [].

    Raises:
        ValueError: Provided path is not point to .json file

    Returns:
        typing.Any: Pytree loaded from JSON
    """

    # Validate that file extension is .json
    extension = str(path).split(".")[-1]

    if extension != "json":
        raise ValueError("File extension must be json")

    if isinstance(path, str):
        path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    data = recursive_parse(data, parse_fn=parse_fn)

    return data


def param_shape_to_dict(x):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            r[k] = {"shape": v.shape}
        return r
    else:
        return x


def is_param_shape(x):
    # Check if it is the dict of ParamShape
    if isinstance(x, dict):
        r = True
        for k, v in x.items():
            r = r and isinstance(v, ParamShape)
        return r
    return False


def save_pytree_to_json(pytree, path: str | Path):
    """Save given pytree to json file, the path must end with extension of .json

    Args:
        pytree (typing.Any): The pytree to save
        path (str | Path): File path to save

    """

    # Convert jax.ndarray
    data = jax.tree.map(
        lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, pytree
    )
    # Convert ParamShape
    data = jax.tree.map(param_shape_to_dict, data, is_leaf=is_param_shape)

    if isinstance(path, str):
        path = Path(path)

    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def from_long_to_wide(preprocessed_df: pd.DataFrame):
    """An experimental function to transform preprocess dataframe to postprocess dataframe.

    Args:
        preprocessed_df (pd.DataFrame): The preprocess dataframe

    Returns:
        pd.DataFrame: The postprocessed dataframe
    """
    # Handle the expectation value using unstack
    expvals_df = preprocessed_df.pivot(
        index="parameters_id",
        columns=["initial_state", "observable"],
        values="expectation_value",  # Note: string, not a list
    )

    # Rename columns using another idiomatic approach
    expvals_df.columns = [
        f"expectation_value/{state}/{obs}" for state, obs in expvals_df.columns
    ]

    # Handle parameters columns
    params_df = (
        preprocessed_df.groupby("parameters_id")
        .first()
        .drop(
            ["expectation_value", "initial_state", "observable"], axis=1, inplace=False
        )
    )

    # Combine with join
    return params_df.join(expvals_df)


def from_wide_to_long_simple(wide_df: pd.DataFrame):
    """
    A more concise version to convert a wide DataFrame back to the long format.
    """
    # Work with the index as a column
    df = wide_df.reset_index()

    # 1. Identify all columns that should NOT be melted (the "id" columns)
    id_vars = [col for col in df.columns if not col.startswith("expectation_value/")]

    # 2. Melt the DataFrame. pd.melt automatically uses all columns NOT in id_vars as value_vars.
    long_df = df.melt(
        id_vars=id_vars, var_name="descriptor", value_name="expectation_value"
    )

    # 3. Split the descriptor and assign new columns in one step
    long_df[["_,", "initial_state", "observable"]] = long_df["descriptor"].str.split(
        "/", expand=True
    )

    # 4. Clean up the DataFrame by dropping temporary columns and sorting
    return (
        long_df.drop(columns=["descriptor", "_,"])
        .sort_values("parameters_id")
        .reset_index(drop=True)
    )
