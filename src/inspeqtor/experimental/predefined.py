import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import typing
from enum import Enum, StrEnum, auto
from functools import partial
import pandas as pd
import pathlib
from .data import (
    QubitInformation,
    ExperimentConfiguration,
    ExperimentData,
    State,
    make_row,
)
from .pulse import (
    BasePulse,
    PulseSequence,
    array_to_list_of_params,
    list_of_params_to_array,
    construct_pulse_sequence_reader,
)
from .typing import ParametersDictType, HamiltonianArgs
from .physics import (
    SignalParameters,
    solver,
    signal_func_v5,
    make_trotterization_whitebox,
)
from .constant import X, Y, Z, default_expectation_values_order
from .decorator import warn_not_tested_function
from .utils import (
    center_location,
    drag_envelope_v2,
    calculate_exp,
    LoadedData,
    prepare_data,
)
from .model import DataConfig


def rotating_transmon_hamiltonian(
    params: HamiltonianArgs,
    t: jnp.ndarray,
    qubit_info: QubitInformation,
    signal: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Rotating frame hamiltonian of the transmon model

    Args:
        params (HamiltonianParameters): The parameter of the pulse for hamiltonian
        t (jnp.ndarray): The time to evaluate the Hamiltonian
        qubit_info (QubitInformation): The information of qubit
        signal (Callable[..., jnp.ndarray]): The pulse signal

    Returns:
        jnp.ndarray: The Hamiltonian
    """
    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    def f3(params, t):
        return a1 * signal(params, t)

    def f_sigma_x(params, t):
        return f3(params, t) * jnp.cos(a0 * t)

    def f_sigma_y(params, t):
        return f3(params, t) * jnp.sin(a0 * t)

    return f_sigma_x(params, t) * X - f_sigma_y(params, t) * Y


def transmon_hamiltonian(
    params: HamiltonianArgs,
    t: jnp.ndarray,
    qubit_info: QubitInformation,
    signal: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Lab frame hamiltonian of the transmon model

    Args:
        params (HamiltonianParameters): The parameter of the pulse for hamiltonian
        t (jnp.ndarray): The time to evaluate the Hamiltonian
        qubit_info (QubitInformation): The information of qubit
        signal (Callable[..., jnp.ndarray]): The pulse signal

    Returns:
        jnp.ndarray: The Hamiltonian
    """

    a0 = 2 * jnp.pi * qubit_info.frequency
    a1 = 2 * jnp.pi * qubit_info.drive_strength

    return ((a0 / 2) * Z) + (a1 * signal(params, t) * X)


def get_envelope(params: ParametersDictType, order: int, total_length: int):
    # Callables
    envelopes = []
    for i in range(order):
        # The center location of each drag pulse in the order
        center_locations = center_location(i + 1, total_length)
        for j, center in enumerate(center_locations):
            envelopes.append(
                drag_envelope_v2(
                    amp=params[f"{i}/{j}/amp"],
                    sigma=params[f"{i}/{j}/sigma"],
                    beta=0,
                    center=center,
                    final_amp=1 / (2 ** (i + 1)),
                )
            )

    def sum_envelopes(t):
        return jnp.real(sum([envelope(t) for envelope in envelopes]))

    real_part_fn = sum_envelopes
    drag_part_fn = jax.grad(real_part_fn)

    return lambda t: real_part_fn(t) + 1j * params["beta"] * drag_part_fn(t)


@dataclass
class DragPulse(BasePulse):
    duration: int
    beta: float
    qubit_drive_strength: float
    dt: float
    amp: float = 0.25

    min_theta: float = 0.0
    max_theta: float = 2 * jnp.pi
    final_amp: float = 1.0

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration)

    def get_bounds(self) -> tuple[ParametersDictType, ParametersDictType]:
        lower = {}
        upper = {}

        lower["theta"] = self.min_theta
        upper["theta"] = self.max_theta

        return lower, upper

    def get_waveform(self, params: ParametersDictType) -> jnp.ndarray:
        return self.get_envelope(params)(self.t_eval)

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        area = (
            params["theta"] / (2 * jnp.pi * self.qubit_drive_strength)
        ) / self.dt  # NOTE: Choice of area is arbitrary e.g. pi pulse
        sigma = (1 * area) / (self.amp * jnp.sqrt(2 * jnp.pi))

        return drag_envelope_v2(
            amp=self.amp,
            sigma=sigma.astype(float),
            beta=self.beta,
            center=self.duration // 2,
            final_amp=self.final_amp,
        )


@dataclass
class MultiDragPulseV3(BasePulse):
    duration: int
    order: int = 1
    amp_bound: list[list[float]] = field(default_factory=list)  # [[0.0, 1.0],]
    sigma_bound: list[list[float]] = field(default_factory=list)  # [[0.1, 5.0],]
    global_beta_bound: list[float] = field(default_factory=list)  # [-2.0, 2.0]

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration, dtype=jnp.float64)

    def get_bounds(self) -> tuple[ParametersDictType, ParametersDictType]:
        lower = {}
        upper = {}

        idx = 0
        for i in range(self.order):
            for j in range(i + 1):
                lower[f"{i}/{j}/amp"] = self.amp_bound[idx][0]
                lower[f"{i}/{j}/sigma"] = self.sigma_bound[idx][0]

                upper[f"{i}/{j}/amp"] = self.amp_bound[idx][1]
                upper[f"{i}/{j}/sigma"] = self.sigma_bound[idx][1]

                idx += 1

        lower["beta"] = self.global_beta_bound[0]
        upper["beta"] = self.global_beta_bound[1]

        return lower, upper

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        return get_envelope(params, self.order, self.duration)


def get_drag_pulse_sequence(
    qubit_info: QubitInformation,
    amp: float = 0.5,  # NOTE: Choice of amplitude is arbitrary
):
    total_length = 320
    dt = 2 / 9

    pulse_sequence = PulseSequence(
        pulses=[
            DragPulse(
                duration=total_length,
                beta=0,
                qubit_drive_strength=qubit_info.drive_strength,
                dt=dt,
                amp=amp,
                min_theta=0.0,
                max_theta=2 * jnp.pi,
                final_amp=1.0,
            )
        ],
        pulse_length_dt=total_length,
    )

    return pulse_sequence


def get_multi_drag_pulse_sequence_v3():
    order = 4
    amp_bounds = list([[0.0, 1.0]] * order)
    order_amp_bound = list(
        [amp_bound for idx, amp_bound in enumerate(amp_bounds) for _ in range(idx + 1)]
    )
    sigma_bounds = [[7.0, 9.0], [5.0, 7.0], [3.0, 5.0], [1.0, 3.0]]
    order_sigma_bound = list(
        [
            sigma_bound
            for idx, sigma_bound in enumerate(sigma_bounds)
            for _ in range(idx + 1)
        ]
    )

    pulse = MultiDragPulseV3(
        duration=80,
        order=order,
        amp_bound=order_amp_bound,
        sigma_bound=order_sigma_bound,
        global_beta_bound=[-2.0, 2.0],
    )

    pulse_sequence = PulseSequence(
        pulse_length_dt=80,
        pulses=[pulse],
    )
    return pulse_sequence


def get_mock_qubit_information() -> QubitInformation:
    return QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0,
        drive_strength=0.1,
    )


def get_mock_prefined_exp_v1(
    sample_size: int = 10,
    shots: int = 1000,
    get_qubit_information_fn: typing.Callable[
        [], QubitInformation
    ] = get_mock_qubit_information,
    get_pulse_sequence_fn: typing.Callable[
        [], PulseSequence
    ] = get_multi_drag_pulse_sequence_v3,
):
    qubit_info = get_qubit_information_fn()
    pulse_sequence = get_pulse_sequence_fn()

    config = ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=default_expectation_values_order,
        parameter_names=pulse_sequence.get_parameter_names(),
        backend_name="fake_ibm_test",
        shots=shots,
        EXPERIMENT_IDENTIFIER="test",
        EXPERIMENT_TAGS=["test"],
        description="Generated for test",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=pulse_sequence.pulse_length_dt,
        instance="inspeqtor/tester",
        sample_size=sample_size,
    )

    return qubit_info, pulse_sequence, config


class SimulationStrategy(Enum):
    IDEAL = "ideal"
    RANDOM = "random"
    SHOT = "shot"
    NOISY = "noisy"


plus_projectors = {
    "X": State.from_label("+", dm=True),
    "Y": State.from_label("r", dm=True),
    "Z": State.from_label("0", dm=True),
}


def calculate_shots_expectation_value(
    key,
    initial_state: jnp.ndarray,
    unitary: jnp.ndarray,
    plus_projector: jnp.ndarray,
    shots: int,
):
    prob = jnp.trace(unitary @ initial_state @ unitary.conj().T @ plus_projector).real

    return jax.random.choice(
        key, jnp.array([1, -1]), shape=(shots,), p=jnp.array([prob, 1 - prob])
    ).mean()


class WhiteboxStrategy(StrEnum):
    ODE = auto()
    TROTTER = auto()


@warn_not_tested_function
def generate_mock_experiment_data(
    key: jnp.ndarray,
    sample_size: int = 10,
    shots: int = 1000,
    strategy: SimulationStrategy = SimulationStrategy.RANDOM,
    hamiltonian_transformers: list[
        typing.Callable[
            [typing.Callable[..., jnp.ndarray]], typing.Callable[..., jnp.ndarray]
        ]
    ] = [],
    get_qubit_information_fn: typing.Callable[
        [], QubitInformation
    ] = get_mock_qubit_information,
    get_pulse_sequence_fn: typing.Callable[
        [], PulseSequence
    ] = get_multi_drag_pulse_sequence_v3,
    max_steps: int = int(2**16),
    method: WhiteboxStrategy = WhiteboxStrategy.ODE,
):
    qubit_info, pulse_sequence, config = get_mock_prefined_exp_v1(
        sample_size=sample_size,
        shots=shots,
        get_pulse_sequence_fn=get_pulse_sequence_fn,
        get_qubit_information_fn=get_qubit_information_fn,
    )

    # Generate mock expectation value
    key, exp_key = jax.random.split(key)

    dt = config.device_cycle_time_ns

    ideal_hamiltonian = partial(
        rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v5(
            get_envelope_transformer(pulse_sequence),
            qubit_info.frequency,
            dt,
        ),
    )

    hamiltonian = ideal_hamiltonian
    for transformer in hamiltonian_transformers:
        hamiltonian = transformer(hamiltonian)

    if method == WhiteboxStrategy.TROTTER:
        whitebox = jax.jit(
            make_trotterization_whitebox(
                hamiltonian=ideal_hamiltonian, pulse_sequence=pulse_sequence, dt=2 / 9
            )
        )

        noisy_simulator = jax.jit(
            make_trotterization_whitebox(
                hamiltonian=hamiltonian, pulse_sequence=pulse_sequence, dt=2 / 9
            )
        )
    else:
        whitebox = jax.jit(
            get_single_qubit_whitebox(
                hamiltonian=ideal_hamiltonian,
                pulse_sequence=pulse_sequence,
                qubit_info=qubit_info,
                dt=dt,
                max_steps=max_steps,
            )
        )

        noisy_simulator = jax.jit(
            get_single_qubit_whitebox(
                hamiltonian=hamiltonian,
                pulse_sequence=pulse_sequence,
                qubit_info=qubit_info,
                dt=dt,
            )
        )

    SHOTS = config.shots

    rows = []
    pulse_params_list = []
    signal_params_list: list[SignalParameters] = []
    pulse_parameters = []
    parameter_structure = pulse_sequence.get_parameter_names()
    for sample_idx in range(config.sample_size):
        key, subkey = jax.random.split(key)
        pulse_params = pulse_sequence.sample_params(subkey)
        pulse_params_list.append(pulse_params)

        signal_param = SignalParameters(pulse_params=pulse_params, phase=0)
        signal_params_list.append(signal_param)

        pulse_parameters.append(
            list_of_params_to_array(pulse_params, parameter_structure)
        )

    pulse_parameters = jnp.array(pulse_parameters)

    unitaries = jax.vmap(noisy_simulator)(pulse_parameters)

    # Calculate the expectation values depending on the strategy
    expectation_values: jnp.ndarray
    expvals = []
    final_unitaries = jnp.array(unitaries)[:, -1, :, :]

    assert final_unitaries.shape == (
        sample_size,
        2,
        2,
    ), f"Final unitaries shape is {final_unitaries.shape}"

    if strategy == SimulationStrategy.RANDOM:
        # Just random expectation values with key
        expectation_values = 2 * (
            jax.random.uniform(exp_key, shape=(config.sample_size, 18)) - (1 / 2)
        )
    elif strategy == SimulationStrategy.IDEAL:
        for exp in default_expectation_values_order:
            expval = calculate_exp(
                final_unitaries, exp.observable_matrix, exp.initial_density_matrix
            )

            expvals.append(expval)

        expectation_values = jnp.transpose(jnp.array(expvals))

    elif strategy == SimulationStrategy.SHOT:
        for exp in default_expectation_values_order:
            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, num=final_unitaries.shape[0])

            expval = jax.vmap(
                calculate_shots_expectation_value, in_axes=(0, None, 0, None, None)
            )(
                sample_keys,
                exp.initial_density_matrix,
                final_unitaries,
                plus_projectors[exp.observable],
                SHOTS,
            )

            expvals.append(expval)

        expectation_values = jnp.transpose(jnp.array(expvals))

    else:
        raise NotImplementedError

    assert expectation_values.shape == (
        sample_size,
        18,
    ), f"Expectation values shape is {expectation_values.shape}"

    for sample_idx in range(config.sample_size):
        for exp_idx, exp in enumerate(default_expectation_values_order):
            row = make_row(
                expectation_value=float(expectation_values[sample_idx, exp_idx]),
                initial_state=exp.initial_state,
                observable=exp.observable,
                parameters_list=pulse_params_list[sample_idx],
                parameters_id=sample_idx,
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    exp_data = ExperimentData(experiment_config=config, preprocess_data=df)

    return (
        exp_data,
        pulse_sequence,
        jnp.array(unitaries),
        signal_params_list,
        noisy_simulator,
        whitebox,
        (ideal_hamiltonian, hamiltonian),
    )


def get_envelope_transformer(pulse_sequence: PulseSequence):
    structure = pulse_sequence.get_parameter_names()

    def array_to_list_of_params_fn(array: jnp.ndarray):
        return array_to_list_of_params(array, structure)

    def get_envelope(params: jnp.ndarray) -> typing.Callable[..., typing.Any]:
        return pulse_sequence.get_envelope(array_to_list_of_params_fn(params))

    return get_envelope


def get_single_qubit_whitebox(
    hamiltonian: typing.Callable[..., jnp.ndarray],
    pulse_sequence: PulseSequence,
    qubit_info: QubitInformation,
    dt: float,
    max_steps: int = int(2**16),
):
    t_eval = jnp.linspace(
        0, pulse_sequence.pulse_length_dt * dt, pulse_sequence.pulse_length_dt
    )

    hamiltonian = partial(
        hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v5(
            get_envelope_transformer(pulse_sequence),
            qubit_info.frequency,
            dt,
        ),
    )

    whitebox = partial(
        solver,
        t_eval=t_eval,
        hamiltonian=hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
        t0=0,
        t1=pulse_sequence.pulse_length_dt * dt,
        max_steps=max_steps,
    )

    return whitebox


def get_single_qubit_rotating_frame_whitebox(
    pulse_sequence: PulseSequence,
    qubit_info: QubitInformation,
    dt: float,
):
    whitebox = get_single_qubit_whitebox(
        rotating_transmon_hamiltonian,
        pulse_sequence,
        qubit_info,
        dt,
    )

    return whitebox


pulse_reader = construct_pulse_sequence_reader(pulses=[DragPulse, MultiDragPulseV3])

hamiltonian_mapper = {
    "transmon_hamiltonian": transmon_hamiltonian,
    "rotating_transmon_hamiltonian": rotating_transmon_hamiltonian,
}


@warn_not_tested_function
def load_data_from_path(
    path: str | pathlib.Path,
    model_path: str | pathlib.Path,
) -> LoadedData:
    if isinstance(path, str):
        path = pathlib.Path(path)

    if isinstance(model_path, str):
        model_path = pathlib.Path(model_path)

    exp_data = ExperimentData.from_folder(path)
    pulse_sequence = pulse_reader(path)

    # Read hamiltonian from data config
    data_config = DataConfig.from_file(model_path)

    whitebox = get_single_qubit_whitebox(
        hamiltonian_mapper[data_config.hamiltonian],
        pulse_sequence,
        exp_data.experiment_config.qubits[0],
        exp_data.experiment_config.device_cycle_time_ns,
    )

    return prepare_data(exp_data, pulse_sequence, whitebox)


def gaussian_envelope(amp, center, sigma):
    def g_fn(t):
        return (amp / (jnp.sqrt(2 * jnp.pi) * sigma)) * jnp.exp(
            -((t - center) ** 2) / (2 * sigma**2)
        )

    return g_fn


@dataclass
class GaussianPulse(BasePulse):
    duration: int
    # beta: float
    qubit_drive_strength: float
    dt: float
    max_amp: float = 0.25

    min_theta: float = 0.0
    max_theta: float = 2 * jnp.pi

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration)

        # This is the correction factor that will cancel the factor in the front of hamiltonian
        self.correction = 2 * jnp.pi * self.qubit_drive_strength * self.dt

        # The standard derivation of Gaussian pulse is keep fixed for the given max_amp
        self.sigma = jnp.sqrt(2 * jnp.pi) / (self.max_amp * self.correction)

        # The center position is set at the center of the duration
        self.center_position = self.duration // 2

    def get_bounds(
        self,
    ) -> tuple[ParametersDictType, ParametersDictType]:
        lower = {}
        upper = {}

        lower["theta"] = self.min_theta
        upper["theta"] = self.max_theta

        return lower, upper

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        # The area of Gaussian to be rotate to,
        area = (
            params["theta"] / self.correction
        )  # NOTE: Choice of area is arbitrary e.g. pi pulse

        return gaussian_envelope(
            amp=area, center=self.center_position, sigma=self.sigma
        )


def get_gaussian_pulse_sequence(
    qubit_info: QubitInformation,
    max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
):
    total_length = 320
    dt = 2 / 9

    pulse_sequence = PulseSequence(
        pulses=[
            GaussianPulse(
                duration=total_length,
                qubit_drive_strength=qubit_info.drive_strength,
                dt=dt,
                max_amp=max_amp,
                min_theta=0.0,
                max_theta=2 * jnp.pi,
            ),
        ],
        pulse_length_dt=total_length,
    )

    return pulse_sequence


def polynomial_feature_map(x: jnp.ndarray, degree: int):
    return jnp.concatenate([x**i for i in range(1, degree + 1)], axis=-1)


def detune_x_hamiltonian(
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    detune: float,
) -> typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
    def detuned_hamiltonian(
        params: HamiltonianArgs,
        t: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return hamiltonian(params, t, *args, **kwargs) + detune * X

    return detuned_hamiltonian
