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
    make_row,
)
from .control import (
    BaseControl,
    ControlSequence,
    array_to_list_of_params,
    list_of_params_to_array,
    construct_control_sequence_reader,
)
from .typing import ParametersDictType, HamiltonianArgs
from .physics import (
    solver,
    signal_func_v5,
    make_trotterization_whitebox,
)
from .constant import X, Y, Z, default_expectation_values_order, plus_projectors
from .utils import (
    center_location,
    drag_envelope_v2,
    LoadedData,
    prepare_data,
    calculate_expectation_values,
    calculate_shots_expectation_value,
)
import itertools


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


def gaussian_envelope(amp, center, sigma):
    def g_fn(t):
        return (amp / (jnp.sqrt(2 * jnp.pi) * sigma)) * jnp.exp(
            -((t - center) ** 2) / (2 * sigma**2)
        )

    return g_fn


@dataclass
class GaussianPulse(BaseControl):
    duration: int
    # beta: float
    qubit_drive_strength: float
    dt: float
    max_amp: float = 0.25

    min_theta: float = 0.0
    max_theta: float = 2 * jnp.pi

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration, dtype=jnp.float_)

        # This is the correction factor that will cancel the factor in the front of hamiltonian
        self.correction = 2 * jnp.pi * self.qubit_drive_strength * self.dt

        # The standard derivation of Gaussian pulse is keep fixed for the given max_amp
        self.sigma = jnp.sqrt(2 * jnp.pi) / (self.max_amp * self.correction)

        # The center position is set at the center of the duration
        self.center_position = self.duration // 2

    def get_bounds(
        self,
    ) -> tuple[ParametersDictType, ParametersDictType]:
        lower: ParametersDictType = {}
        upper: ParametersDictType = {}

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


def get_gaussian_control_sequence(
    qubit_info: QubitInformation,
    max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
):
    """Get predefined Gaussian control sequence with single Gaussian pulse.

    Args:
        qubit_info (QubitInformation): Qubit information
        max_amp (float, optional): The maximum amplitude. Defaults to 0.5.

    Returns:
        PulseSequence: Control sequence instance
    """
    total_length = 320
    dt = 2 / 9

    control_sequence = ControlSequence(
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

    return control_sequence


@dataclass
class TwoAxisGaussianPulse(BaseControl):
    duration: int
    qubit_drive_strength: float
    dt: float
    max_amp: float = 0.25

    # Rotation angles for both axes
    min_theta_x: float = 0.0
    max_theta_x: float = 2 * jnp.pi
    min_theta_y: float = 0.0
    max_theta_y: float = 2 * jnp.pi

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration, dtype=jnp.float_)

        # Correction factor that will cancel the factor in the front of hamiltonian
        self.correction = 2 * jnp.pi * self.qubit_drive_strength * self.dt

        # The standard deviation of Gaussian pulse is kept fixed for the given max_amp
        self.sigma = jnp.sqrt(2 * jnp.pi) / (self.max_amp * self.correction)

        # The center position is set at the center of the duration
        self.center_position = self.duration // 2

    def get_bounds(
        self,
    ) -> tuple[ParametersDictType, ParametersDictType]:
        lower: ParametersDictType = {}
        upper: ParametersDictType = {}

        # Bounds for X-axis rotation
        lower["theta_x"] = self.min_theta_x
        upper["theta_x"] = self.max_theta_x

        # Bounds for Y-axis rotation
        lower["theta_y"] = self.min_theta_y
        upper["theta_y"] = self.max_theta_y

        return lower, upper

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        # Calculate areas for both axes
        area_x = params["theta_x"] / self.correction
        area_y = params["theta_y"] / self.correction

        def envelope_fn(t):
            # Gaussian envelope for both axes
            x_axis = gaussian_envelope(
                amp=area_x, center=self.center_position, sigma=self.sigma
            )(t)

            y_axis = gaussian_envelope(
                amp=area_y, center=self.center_position, sigma=self.sigma
            )(t)

            # Return complex envelope with x and y components
            return x_axis + 1j * y_axis

        return envelope_fn


def get_two_axis_gaussian_control_sequence(
    qubit_info: QubitInformation,
    max_amp: float = 0.5,
):
    """Get predefined two-axis Gaussian control sequence.

    Args:
        qubit_info (QubitInformation): Qubit information
        max_amp (float, optional): The maximum amplitude. Defaults to 0.5.

    Returns:
        PulseSequence: Control sequence instance
    """
    total_length = 320
    dt = 2 / 9

    control_sequence = ControlSequence(
        pulses=[
            TwoAxisGaussianPulse(
                duration=total_length,
                qubit_drive_strength=qubit_info.drive_strength,
                dt=dt,
                max_amp=max_amp,
                min_theta_x=-2 * jnp.pi,
                max_theta_x=2 * jnp.pi,
                min_theta_y=-2 * jnp.pi,
                max_theta_y=2 * jnp.pi,
            ),
        ],
        pulse_length_dt=total_length,
    )

    return control_sequence


@dataclass
class DragPulseV2(BaseControl):
    duration: int
    qubit_drive_strength: float
    dt: float
    max_amp: float = 0.25

    min_theta: float = 0.0
    max_theta: float = 2 * jnp.pi

    min_beta: float = 0.0
    max_beta: float = 2.0

    def __post_init__(self):
        self.gaussian_pulse = GaussianPulse(
            duration=self.duration,
            qubit_drive_strength=self.qubit_drive_strength,
            dt=self.dt,
            max_amp=self.max_amp,
            min_theta=self.min_theta,
            max_theta=self.max_theta,
        )
        self.t_eval = self.gaussian_pulse.t_eval

    def get_bounds(
        self,
    ) -> tuple[ParametersDictType, ParametersDictType]:
        lower, upper = self.gaussian_pulse.get_bounds()

        lower["beta"] = self.min_beta
        upper["beta"] = self.max_beta

        return lower, upper

    def get_envelope(
        self, params: ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        # The area of Gaussian to be rotate to,
        area = (
            params["theta"] / self.gaussian_pulse.correction
        )  # NOTE: Choice of area is arbitrary e.g. pi pulse

        def real_component(t):
            return gaussian_envelope(
                amp=area,
                center=self.gaussian_pulse.center_position,
                sigma=self.gaussian_pulse.sigma,
            )(t)

        def envelope_fn(t):
            return real_component(t) + 1j * params["beta"] * jax.grad(real_component)(t)

        return envelope_fn


def get_drag_pulse_v2_sequence(
    qubit_info: QubitInformation,
    max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
):
    """Get predefined DRAG control sequence with single DRAG pulse.

    Args:
        qubit_info (QubitInformation): Qubit information
        max_amp (float, optional): The maximum amplitude. Defaults to 0.5.

    Returns:
        PulseSequence: Control sequence instance
    """
    total_length = 320
    dt = 2 / 9

    control_sequence = ControlSequence(
        pulses=[
            DragPulseV2(
                duration=total_length,
                qubit_drive_strength=qubit_info.drive_strength,
                dt=dt,
                max_amp=max_amp,
                min_theta=0.0,
                max_theta=2 * jnp.pi,
                min_beta=-2.0,
                max_beta=2.0,
            ),
        ],
        pulse_length_dt=total_length,
    )

    return control_sequence


@dataclass
class DragPulse(BaseControl):
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
        lower: ParametersDictType = {}
        upper: ParametersDictType = {}

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


def get_drag_control_sequence(
    qubit_info: QubitInformation,
    amp: float = 0.5,  # NOTE: Choice of amplitude is arbitrary
):
    total_length = 320
    dt = 2 / 9

    control_sequence = ControlSequence(
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

    return control_sequence


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
class MultiDragPulseV3(BaseControl):
    duration: int
    order: int = 1
    amp_bound: list[list[float]] = field(default_factory=list)  # [[0.0, 1.0],]
    sigma_bound: list[list[float]] = field(default_factory=list)  # [[0.1, 5.0],]
    global_beta_bound: list[float] = field(default_factory=list)  # [-2.0, 2.0]

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration, dtype=jnp.float64)

    def get_bounds(self) -> tuple[ParametersDictType, ParametersDictType]:
        lower: ParametersDictType = {}
        upper: ParametersDictType = {}

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


def get_multi_drag_control_sequence_v3():
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

    control_sequence = ControlSequence(
        pulse_length_dt=80,
        pulses=[pulse],
    )
    return control_sequence


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
    get_control_sequence_fn: typing.Callable[
        [], ControlSequence
    ] = get_multi_drag_control_sequence_v3,
):
    qubit_info = get_qubit_information_fn()
    control_sequence = get_control_sequence_fn()

    config = ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=default_expectation_values_order,
        parameter_names=control_sequence.get_parameter_names(),
        backend_name="fake_ibm_test",
        shots=shots,
        EXPERIMENT_IDENTIFIER="test",
        EXPERIMENT_TAGS=["test"],
        description="Generated for test",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=control_sequence.pulse_length_dt,
        instance="inspeqtor/tester",
        sample_size=sample_size,
    )

    return qubit_info, control_sequence, config


class SimulationStrategy(Enum):
    IDEAL = "ideal"
    RANDOM = "random"
    SHOT = "shot"
    NOISY = "noisy"


class WhiteboxStrategy(StrEnum):
    ODE = auto()
    TROTTER = auto()


#
def generate_experimental_data(
    key: jnp.ndarray,
    hamiltonian: typing.Callable[..., jnp.ndarray],
    sample_size: int = 10,
    shots: int = 1000,
    strategy: SimulationStrategy = SimulationStrategy.RANDOM,
    get_qubit_information_fn: typing.Callable[
        [], QubitInformation
    ] = get_mock_qubit_information,
    get_control_sequence_fn: typing.Callable[
        [], ControlSequence
    ] = get_multi_drag_control_sequence_v3,
    max_steps: int = int(2**16),
    method: WhiteboxStrategy = WhiteboxStrategy.ODE,
    trotter_steps: int = 1000,
) -> tuple[
    ExperimentData,
    ControlSequence,
    jnp.ndarray,
    typing.Callable[[jnp.ndarray], jnp.ndarray],
]:
    """Generate simulated dataset

    Args:
        key (jnp.ndarray): Random key
        hamiltonian (typing.Callable[..., jnp.ndarray]): Total Hamiltonian of the device
        sample_size (int, optional): Sample size of the control parameters. Defaults to 10.
        shots (int, optional): Number of shots used to estimate expectation value, will be used if `SimulationStrategy` is `SHOT`, otherwise ignored. Defaults to 1000.
        strategy (SimulationStrategy, optional): Simulation strategy. Defaults to SimulationStrategy.RANDOM.
        get_qubit_information_fn (typing.Callable[ [], QubitInformation ], optional): Function that return qubit information. Defaults to get_mock_qubit_information.
        get_control_sequence_fn (typing.Callable[ [], PulseSequence ], optional): Function that return control sequence. Defaults to get_multi_drag_control_sequence_v3.
        max_steps (int, optional): Maximum step of solver. Defaults to int(2**16).
        method (WhiteboxStrategy, optional): Unitary solver method. Defaults to WhiteboxStrategy.ODE.

    Raises:
        NotImplementedError: Not support strategy

    Returns:
        tuple[ExperimentData, PulseSequence, jnp.ndarray, typing.Callable[[jnp.ndarray], jnp.ndarray]]: tuple of (1) Experiment data, (2) Pulse sequence, (3) Noisy unitary, (4) Noisy solver
    """
    qubit_info, control_sequence, config = get_mock_prefined_exp_v1(
        sample_size=sample_size,
        shots=shots,
        get_control_sequence_fn=get_control_sequence_fn,
        get_qubit_information_fn=get_qubit_information_fn,
    )

    # Generate mock expectation value
    key, exp_key = jax.random.split(key)

    dt = config.device_cycle_time_ns

    if method == WhiteboxStrategy.TROTTER:
        noisy_simulator = jax.jit(
            make_trotterization_whitebox(
                hamiltonian=hamiltonian,
                control_sequence=control_sequence,
                dt=dt,
                trotter_steps=trotter_steps,
            )
        )
    else:
        t_eval = jnp.linspace(
            0, control_sequence.pulse_length_dt * dt, control_sequence.pulse_length_dt
        )
        noisy_simulator = jax.jit(
            partial(
                solver,
                t_eval=t_eval,
                hamiltonian=hamiltonian,
                y0=jnp.eye(2, dtype=jnp.complex64),
                t0=0,
                t1=control_sequence.pulse_length_dt * dt,
                max_steps=max_steps,
            )
        )

    control_params_list = []
    parameter_structure = control_sequence.get_parameter_names()
    num_parameters = len(list(itertools.chain.from_iterable(parameter_structure)))
    # control_params: list[jnp.ndarray] = []
    control_params = jnp.empty(shape=(sample_size, num_parameters))
    for control_idx in range(config.sample_size):
        key, subkey = jax.random.split(key)
        pulse_params = control_sequence.sample_params(subkey)
        control_params_list.append(pulse_params)

        # control_params.append(
        #     list_of_params_to_array(pulse_params, parameter_structure)
        # )
        control_params = control_params.at[control_idx].set(
            list_of_params_to_array(pulse_params, parameter_structure)
        )

    # control_params = jnp.array(control_params)

    unitaries = jax.vmap(noisy_simulator)(control_params)
    SHOTS = config.shots

    # Calculate the expectation values depending on the strategy
    unitaries_f = jnp.asarray(unitaries)[:, -1, :, :]
    # expectation_values = jnp.zeros((config.sample_size, 18))

    assert unitaries_f.shape == (
        sample_size,
        2,
        2,
    ), f"Final unitaries shape is {unitaries_f.shape}"

    if strategy == SimulationStrategy.RANDOM:
        # Just random expectation values with key
        expectation_values = 2 * (
            jax.random.uniform(exp_key, shape=(config.sample_size, 18)) - (1 / 2)
        )
    elif strategy == SimulationStrategy.IDEAL:
        expectation_values = calculate_expectation_values(unitaries_f)

    elif strategy == SimulationStrategy.SHOT:
        expectation_values = jnp.zeros((config.sample_size, 18))
        for idx, exp in enumerate(default_expectation_values_order):
            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, num=unitaries_f.shape[0])

            expval = jax.vmap(
                calculate_shots_expectation_value, in_axes=(0, None, 0, None, None)
            )(
                sample_keys,
                exp.initial_density_matrix,
                unitaries_f,
                plus_projectors[exp.observable],
                SHOTS,
            )

            expectation_values = expectation_values.at[..., idx].set(expval)

    else:
        raise NotImplementedError

    assert expectation_values.shape == (
        sample_size,
        18,
    ), f"Expectation values shape is {expectation_values.shape}"

    rows = []
    for sample_idx in range(config.sample_size):
        for exp_idx, exp in enumerate(default_expectation_values_order):
            row = make_row(
                expectation_value=float(expectation_values[sample_idx, exp_idx]),
                initial_state=exp.initial_state,
                observable=exp.observable,
                parameters_list=control_params_list[sample_idx],
                parameters_id=sample_idx,
            )

            rows.append(row)

    df = pd.DataFrame(rows)

    exp_data = ExperimentData(experiment_config=config, preprocess_data=df)

    return (
        exp_data,
        control_sequence,
        jnp.array(unitaries),
        noisy_simulator,
    )


def get_envelope_transformer(control_sequence: ControlSequence):
    """Generate get_envelope function with control parameter array as an input instead of list form

    Args:
        control_sequence (PulseSequence): Control seqence instance

    Returns:
        typing.Callable[[jnp.ndarray], typing.Any]: Transformed get envelope function
    """
    structure = control_sequence.get_parameter_names()

    def array_to_list_of_params_fn(array: jnp.ndarray):
        return array_to_list_of_params(array, structure)

    def get_envelope(params: jnp.ndarray) -> typing.Callable[..., typing.Any]:
        return control_sequence.get_envelope(array_to_list_of_params_fn(params))

    return get_envelope


def get_single_qubit_whitebox(
    hamiltonian: typing.Callable[..., jnp.ndarray],
    control_sequence: ControlSequence,
    qubit_info: QubitInformation,
    dt: float,
    max_steps: int = int(2**16),
) -> typing.Callable[[jnp.ndarray], jnp.ndarray]:
    """Generate single qubit whitebox

    Args:
        hamiltonian (typing.Callable[..., jnp.ndarray]): Hamiltonian
        control_sequence (PulseSequence): Control sequence instance
        qubit_info (QubitInformation): Qubit information
        dt (float): Duration of 1 timestep in nanosecond
        max_steps (int, optional): Maximum steps of solver. Defaults to int(2**16).

    Returns:
        typing.Callable[[jnp.ndarray], jnp.ndarray]: Whitebox with ODE solver
    """
    t_eval = jnp.linspace(
        0, control_sequence.pulse_length_dt * dt, control_sequence.pulse_length_dt
    )

    hamiltonian = partial(
        hamiltonian,
        qubit_info=qubit_info,
        signal=signal_func_v5(
            get_envelope_transformer(control_sequence),
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
        t1=control_sequence.pulse_length_dt * dt,
        max_steps=max_steps,
    )

    return whitebox


def get_single_qubit_rotating_frame_whitebox(
    control_sequence: ControlSequence,
    qubit_info: QubitInformation,
    dt: float,
) -> typing.Callable[[jnp.ndarray], jnp.ndarray]:
    """Generate single qubit whitebox with rotating transmon hamiltonian

    Args:
        control_sequence (PulseSequence): Control sequence
        qubit_info (QubitInformation): Qubit information
        dt (float): Duration of 1 timestep in nanosecond

    Returns:
        typing.Callable[[jnp.ndarray], jnp.ndarray]: Whitebox with ODE solver and rotating transmon hamiltonian
    """
    whitebox = get_single_qubit_whitebox(
        rotating_transmon_hamiltonian,
        control_sequence,
        qubit_info,
        dt,
    )

    return whitebox


default_pulse_reader = construct_control_sequence_reader(
    pulses=[
        DragPulse,
        MultiDragPulseV3,
        GaussianPulse,
        DragPulseV2,
        TwoAxisGaussianPulse,
    ]
)


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


class HamiltonianEnum(StrEnum):
    transmon_hamiltonian = auto()
    rotating_transmon_hamiltonian = auto()


@dataclass
class HamiltonianSpec:
    method: WhiteboxStrategy
    hamiltonian_enum: HamiltonianEnum = HamiltonianEnum.rotating_transmon_hamiltonian
    # For Trotterization
    trotter_steps: int = 1000
    # For ODE sovler
    max_steps = int(2**16)

    def get_hamiltonian_fn(self):
        if self.hamiltonian_enum == HamiltonianEnum.rotating_transmon_hamiltonian:
            return rotating_transmon_hamiltonian
        elif self.hamiltonian_enum == HamiltonianEnum.transmon_hamiltonian:
            return transmon_hamiltonian
        else:
            raise ValueError(f"Unsupport Hamiltonian: {self.hamiltonian_enum}")

    def get_solver(
        self,
        control_sequence: ControlSequence,
        qubit_info: QubitInformation,
        dt: float,
    ):
        if self.method == WhiteboxStrategy.TROTTER:
            hamiltonian = partial(
                self.get_hamiltonian_fn(),
                qubit_info=qubit_info,
                signal=signal_func_v5(
                    get_envelope=get_envelope_transformer(
                        control_sequence=control_sequence
                    ),
                    drive_frequency=qubit_info.frequency,
                    dt=dt,
                ),
            )

            whitebox = make_trotterization_whitebox(
                hamiltonian=hamiltonian,
                control_sequence=control_sequence,
                dt=dt,
                trotter_steps=self.trotter_steps,
            )
            return whitebox
        elif self.method == WhiteboxStrategy.ODE:
            return get_single_qubit_whitebox(
                self.get_hamiltonian_fn(),
                control_sequence,
                qubit_info,
                dt,
                self.max_steps,
            )
        else:
            raise ValueError("Unsupport method")


def load_data_from_path(
    path: str | pathlib.Path,
    hamiltonian_spec: HamiltonianSpec,
    pulse_reader=default_pulse_reader,
) -> LoadedData:
    exp_data = ExperimentData.from_folder(path)
    control_sequence = pulse_reader(path)

    qubit_info = exp_data.experiment_config.qubits[0]
    dt = exp_data.experiment_config.device_cycle_time_ns

    whitebox = hamiltonian_spec.get_solver(control_sequence, qubit_info, dt)

    return prepare_data(exp_data, control_sequence, whitebox)


def save_data_to_path(
    path: str | pathlib.Path,
    experiment_data: ExperimentData,
    control_sequence: ControlSequence,
):
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    experiment_data.save_to_folder(path)
    control_sequence.to_file(path)

    return None
