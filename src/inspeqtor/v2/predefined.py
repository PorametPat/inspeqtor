import jax
import jax.numpy as jnp
from functools import partial
import pathlib
import typing
import polars as pl
import numpy as np
from flax.traverse_util import flatten_dict

from inspeqtor.v2.data import (
    ExperimentConfiguration,
    ExperimentalData,
    ExpectationValue,
    get_complete_expectation_values,
    QubitInformation,
)
from inspeqtor.v2.control import (
    ControlSequence,
    get_envelope_transformer,
    construct_control_sequence_reader,
    ravel_unravel_fn,
)
from inspeqtor.v2.utils import (
    SyntheticDataModel,
    prepare_data,
    LoadedData,
    calculate_expectation_values,
    shot_quantum_device,
)
from inspeqtor.experimental.predefined import (
    DragPulse,
    DragPulseV2,
    MultiDragPulseV3,
    GaussianPulse,
    TwoAxisGaussianPulse,
    auto_rotating_frame_hamiltonian,
    transmon_hamiltonian,
    HamiltonianSpec,
    SimulationStrategy,
    get_mock_qubit_information,
    WhiteboxStrategy,
)
from inspeqtor.experimental.constant import Z
from inspeqtor.experimental.physics import (
    signal_func_v5,
    make_trotterization_solver,
    solver,
)
from inspeqtor.experimental.visualization import format_expectation_values


def get_drag_pulse_v2_sequence(
    qubit_info_drive_strength: float,
    max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
    min_theta=0.0,
    max_theta=2 * jnp.pi,
    min_beta=-2.0,
    max_beta=2.0,
    dt=2 / 9,
):
    """Get predefined DRAG control sequence with single DRAG pulse.

    Args:
        qubit_info (QubitInformation): Qubit information
        max_amp (float, optional): The maximum amplitude. Defaults to 0.5.

    Returns:
        ControlSequence: Control sequence instance
    """
    total_length = 320
    control_sequence = ControlSequence(
        controls={
            "0": DragPulseV2(
                duration=total_length,
                qubit_drive_strength=qubit_info_drive_strength,
                dt=dt,
                max_amp=max_amp,
                min_theta=min_theta,
                max_theta=max_theta,
                min_beta=min_beta,
                max_beta=max_beta,
            ),
        },
        total_dt=total_length,
    )

    return control_sequence


def get_predefined_data_model_m1(
    detune: float = 0.0001, get_envelope_transformer=get_envelope_transformer
):
    dt = 2 / 9
    real_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0,
        drive_strength=0.1,
    )
    # The drive frequenct is detune by .01%

    characterized_qubit_info = QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0 * (1 + detune),
        drive_strength=0.1,
    )

    control_seq = get_drag_pulse_v2_sequence(
        qubit_info_drive_strength=characterized_qubit_info.drive_strength,
        min_beta=-5.0,
        max_beta=5.0,
        dt=dt,
    )

    signal_fn = signal_func_v5(
        get_envelope=get_envelope_transformer(control_seq),
        drive_frequency=characterized_qubit_info.frequency,
        dt=dt,
    )
    hamiltonian = partial(
        transmon_hamiltonian, qubit_info=real_qubit_info, signal=signal_fn
    )
    frame = (jnp.pi * characterized_qubit_info.frequency) * Z
    hamiltonian = auto_rotating_frame_hamiltonian(hamiltonian, frame=frame)

    TROTTER_STEPS = 10_000

    solver = make_trotterization_solver(
        hamiltonian=hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    ideal_hamiltonian = partial(
        transmon_hamiltonian,
        qubit_info=characterized_qubit_info,
        signal=signal_fn,  # Already used the characterized_qubit
    )
    ideal_hamiltonian = auto_rotating_frame_hamiltonian(ideal_hamiltonian, frame=frame)

    whitebox = make_trotterization_solver(
        hamiltonian=ideal_hamiltonian,
        total_dt=control_seq.total_dt,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )

    return SyntheticDataModel(
        control_sequence=control_seq,
        qubit_information=characterized_qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=hamiltonian,
        solver=solver,
        quantum_device=None,
        whitebox=whitebox,
    )


predefined_controls = [
    DragPulse,
    MultiDragPulseV3,
    GaussianPulse,
    DragPulseV2,
    TwoAxisGaussianPulse,
]

default_control_reader = construct_control_sequence_reader(controls=predefined_controls)


def load_data_from_path(
    path: str | pathlib.Path,
    hamiltonian_spec: HamiltonianSpec,
    control_reader=default_control_reader,
) -> LoadedData:
    """Load and prepare the experimental data from given path and hamiltonian spec.

    Args:
        path (str | pathlib.Path): The path to the folder that contain experimental data.
        hamiltonian_spec (HamiltonianSpec): The specification of the Hamiltonian
        control_reader (typing.Any, optional): _description_. Defaults to default_control_reader.

    Returns:
        LoadedData: The object contatin necessary information for device characterization.
    """
    exp_data = ExperimentalData.from_folder(path)
    control_sequence = control_reader(path)

    qubit_info = exp_data.config.qubits[0]
    dt = exp_data.config.device_cycle_time_ns

    whitebox = hamiltonian_spec.get_solver(
        control_sequence,
        qubit_info,
        dt,
        get_envelope_transformer=get_envelope_transformer,
    )

    return prepare_data(exp_data, control_sequence, whitebox)


def save_data_to_path(
    path: str | pathlib.Path,
    experiment_data: ExperimentalData,
    control_sequence: ControlSequence,
):
    """Save the experimental data to the path

    Args:
        path (str | pathlib.Path): The path to folder to save the experimental data
        experiment_data (ExperimentData): The experimental data object
        control_sequence (ControlSequence): The control sequence that used to create the experimental data.

    Returns:
        None:
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    experiment_data.save_to_folder(path)
    control_sequence.to_file(path)


def generate_single_qubit_experimental_data(
    key: jnp.ndarray,
    hamiltonian: typing.Callable[..., jnp.ndarray],
    sample_size: int = 10,
    shots: int = 1000,
    strategy: SimulationStrategy = SimulationStrategy.SHOT,
    qubit_inforamtion: QubitInformation = get_mock_qubit_information(),
    control_sequence: ControlSequence = get_drag_pulse_v2_sequence(
        get_mock_qubit_information().drive_strength
    ),
    max_steps: int = int(2**16),
    method: WhiteboxStrategy = WhiteboxStrategy.ODE,
    trotter_steps: int = 1000,
    expectation_value_receipt: list[ExpectationValue] = get_complete_expectation_values(
        1
    ),
) -> tuple[
    ExperimentalData,
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
        get_control_sequence_fn (typing.Callable[ [], ControlSequence ], optional): Function that return control sequence. Defaults to get_multi_drag_control_sequence_v3.
        max_steps (int, optional): Maximum step of solver. Defaults to int(2**16).
        method (WhiteboxStrategy, optional): Unitary solver method. Defaults to WhiteboxStrategy.ODE.

    Raises:
        NotImplementedError: Not support strategy

    Returns:
        tuple[ExperimentData, ControlSequence, jnp.ndarray, typing.Callable[[jnp.ndarray], jnp.ndarray]]: tuple of (1) Experiment data, (2) Pulse sequence, (3) Noisy unitary, (4) Noisy solver
    """
    experiment_config = ExperimentConfiguration(
        qubits=[qubit_inforamtion],
        expectation_values_order=get_complete_expectation_values(1),
        parameter_structure=control_sequence.get_structure(),
        backend_name="stardust",
        sample_size=sample_size,
        shots=shots,
        EXPERIMENT_IDENTIFIER="0001",
        EXPERIMENT_TAGS=["test", "test2"],
        description="This is a test experiment",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=control_sequence.total_dt,
        instance="open",
    )

    # Generate mock expectation value
    key, exp_key = jax.random.split(key)

    dt = experiment_config.device_cycle_time_ns

    if method == WhiteboxStrategy.TROTTER:
        noisy_simulator = jax.jit(
            make_trotterization_solver(
                hamiltonian=hamiltonian,
                total_dt=control_sequence.total_dt,
                dt=dt,
                trotter_steps=trotter_steps,
                y0=jnp.eye(2, dtype=jnp.complex128),
            )
        )
    else:
        t_eval = jnp.linspace(
            0, control_sequence.total_dt * dt, control_sequence.total_dt
        )
        noisy_simulator = jax.jit(
            partial(
                solver,
                t_eval=t_eval,
                hamiltonian=hamiltonian,
                y0=jnp.eye(2, dtype=jnp.complex64),
                t0=0,
                t1=control_sequence.total_dt * dt,
                max_steps=max_steps,
            )
        )

    key, sample_key = jax.random.split(key)

    ravel_fn, _ = ravel_unravel_fn(control_sequence)
    # Sample the parameter by vectorization.
    params_dict = jax.vmap(control_sequence.sample_params)(
        jax.random.split(sample_key, experiment_config.sample_size)
    )
    # Prepare parameter in single line
    control_params = jax.vmap(ravel_fn)(params_dict)

    unitaries = jax.vmap(noisy_simulator)(control_params)
    SHOTS = experiment_config.shots

    # Calculate the expectation values depending on the strategy
    unitaries_f = jnp.asarray(unitaries)[:, -1, :, :]

    assert unitaries_f.shape == (
        sample_size,
        2,
        2,
    ), f"Final unitaries shape is {unitaries_f.shape}"

    if strategy == SimulationStrategy.RANDOM:
        # Just random expectation values with key
        expectation_values = 2 * (
            jax.random.uniform(exp_key, shape=(experiment_config.sample_size, 18))
            - (1 / 2)
        )
    elif strategy == SimulationStrategy.IDEAL:
        expectation_values = calculate_expectation_values(unitaries_f)

    elif strategy == SimulationStrategy.SHOT:
        key, sample_key = jax.random.split(key)
        # The `shot_quantum_device` function will re-calculate the unitary
        expectation_values = shot_quantum_device(
            sample_key,
            control_params,
            noisy_simulator,
            SHOTS,
            expectation_value_receipt,
        )
    else:
        raise NotImplementedError

    assert expectation_values.shape == (
        sample_size,
        18,
    ), f"Expectation values shape is {expectation_values.shape}"

    param_df = pl.DataFrame(
        jax.tree.map(lambda x: np.array(x), flatten_dict(params_dict, sep="/"))
    ).with_row_index("parameter_id")

    obs_df = pl.DataFrame(
        jax.tree.map(
            lambda x: np.array(x),
            flatten_dict(format_expectation_values(expectation_values.T), sep="/"),
        )
    ).with_row_index("parameter_id")

    exp_data = ExperimentalData(experiment_config, param_df, obs_df)

    return (
        exp_data,
        control_sequence,
        jnp.array(unitaries),
        noisy_simulator,
    )
