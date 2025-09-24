import jax.numpy as jnp
from dataclasses import dataclass, field, asdict
import typing
from datetime import datetime
from pathlib import Path
import json
import polars as pl

from ..experimental.data import QubitInformation, ExpectationValue


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


@dataclass
class ExperimentalData:
    config: ExperimentConfiguration
    parameter_dataframe: pl.DataFrame
    observed_dataframe: pl.DataFrame

    def __post_init__(self):
        self.validate()

    def validate(self):
        assert "parameter_id" in self.parameter_dataframe
        assert "parameter_id" in self.observed_dataframe

        assert self.parameter_dataframe["parameter_id"].unique().sort().equals(
            self.observed_dataframe["parameter_id"].unique().sort()
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
