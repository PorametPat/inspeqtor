import jax.numpy as jnp
from functools import partial
import pathlib

from inspeqtor.v2.data import ExperimentalData
from inspeqtor.v2.control import (
    ControlSequence,
    get_envelope_transformer,
    construct_control_sequence_reader,
)
from inspeqtor.v2.utils import SyntheticDataModel, prepare_data, LoadedData
from inspeqtor.experimental.predefined import (
    DragPulse,
    DragPulseV2,
    MultiDragPulseV3,
    GaussianPulse,
    TwoAxisGaussianPulse,
    auto_rotating_frame_hamiltonian,
    transmon_hamiltonian,
    HamiltonianSpec,
)
from inspeqtor.experimental.constant import Z
from inspeqtor.experimental.physics import signal_func_v5, make_trotterization_solver
from inspeqtor.experimental.data import QubitInformation


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
