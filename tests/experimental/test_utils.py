import jax
import jax.numpy as jnp
from inspeqtor.experimental.utils import dataloader
import inspeqtor.experimental as sq
from functools import partial
import pathlib


def get_data_model():
    dt = 2 / 9
    real_qubit_info = sq.data.QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0,
        drive_strength=0.1,
    )
    # The drive frequenct is detune by .01%
    detune = 0.0001
    characterized_qubit_info = sq.data.QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.2,
        frequency=5.0 * (1 + detune),
        drive_strength=0.1,
    )

    control_seq = sq.predefined.get_drag_pulse_v2_sequence(
        qubit_info_drive_strength=characterized_qubit_info.drive_strength,
        min_beta=-5.0,
        max_beta=5.0,
        dt=dt,
    )

    signal_fn = sq.physics.signal_func_v5(
        get_envelope=sq.predefined.get_envelope_transformer(control_seq),
        drive_frequency=characterized_qubit_info.frequency,
        dt=dt,
    )
    hamiltonian = partial(
        sq.predefined.transmon_hamiltonian, qubit_info=real_qubit_info, signal=signal_fn
    )
    frame = (jnp.pi * characterized_qubit_info.frequency) * sq.constant.Z
    hamiltonian = sq.physics.auto_rotating_frame_hamiltonian(hamiltonian, frame=frame)

    solver = sq.physics.make_trotterization_solver(
        hamiltonian=hamiltonian,
        control_sequence=control_seq,
        dt=dt,
        trotter_steps=10_000,
    )

    ideal_hamiltonian = partial(
        sq.predefined.transmon_hamiltonian,
        qubit_info=characterized_qubit_info,
        signal=signal_fn,  # Already used the characterized_qubit
    )
    ideal_hamiltonian = sq.physics.auto_rotating_frame_hamiltonian(
        ideal_hamiltonian, frame=frame
    )

    return sq.utils.SyntheticDataModel(
        control_sequence=control_seq,
        qubit_information=characterized_qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=hamiltonian,
        solver=solver,
        quantum_device=None,
        whitebox=None,
    )


def test_dataloader(DATA_SIZE: int = 100, BATCH_SIZE: int = 15, NUM_EPOCHS: int = 10):
    train_key = jax.random.key(0)

    x_mock = jnp.linspace(0, 10, DATA_SIZE).reshape(-1, 1)
    y_mock = jnp.sin(x_mock)

    # Expected number of batches per epoch
    num_batches = x_mock.shape[0] // BATCH_SIZE
    if x_mock.shape[0] % BATCH_SIZE != 0:
        num_batches += 1

    expected_final_batch_idx = num_batches - 1
    expected_step = num_batches * NUM_EPOCHS - 1

    step = 0
    batch_idx = 0
    is_last_batch = True
    epoch_idx = 0

    for (step, batch_idx, is_last_batch, epoch_idx), (x_batch, y_batch) in dataloader(
        (x_mock, y_mock), batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, key=train_key
    ):
        print(
            f"step: {step}, batch_idx: {batch_idx}, is_last_batch: {is_last_batch}, epoch_idx: {epoch_idx}, x_batch: {x_batch.shape}, y_batch: {y_batch.shape}"
        )

    assert step == expected_step
    assert batch_idx == expected_final_batch_idx
    assert is_last_batch
    assert epoch_idx == NUM_EPOCHS - 1


def test_generate_synthetic_dataset(tmp_path: pathlib.Path):
    data_model = get_data_model()
    TROTTER_STEPS = 1000
    TROTTERIZATION = True

    exp_data, _, _, _ = sq.predefined.generate_experimental_data(
        key=jax.random.key(0),
        hamiltonian=data_model.total_hamiltonian,
        sample_size=10,
        strategy=sq.predefined.SimulationStrategy.SHOT,
        get_qubit_information_fn=lambda: data_model.qubit_information,
        get_control_sequence_fn=lambda: data_model.control_sequence,
        method=sq.predefined.WhiteboxStrategy.TROTTER,
        trotter_steps=TROTTER_STEPS,
    )

    exp_data.experiment_config.additional_info["TROTTERIZATION"] = TROTTERIZATION
    exp_data.experiment_config.additional_info["TROTTER_STEPS"] = TROTTER_STEPS

    path = tmp_path / "test"
    path.mkdir(parents=True, exist_ok=True)
    sq.predefined.save_data_to_path(path, exp_data, data_model.control_sequence)

    loaded_data = sq.predefined.load_data_from_path(
        path,
        hamiltonian_spec=sq.predefined.HamiltonianSpec(
            method=sq.predefined.WhiteboxStrategy.TROTTER
            if TROTTERIZATION
            else sq.predefined.WhiteboxStrategy.ODE,
            trotter_steps=TROTTER_STEPS,
        ),
    )

    assert loaded_data.experiment_data == exp_data
