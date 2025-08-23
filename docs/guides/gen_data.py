import jax
import jax.numpy as jnp
import inspeqtor.experimental as sq

# --8<-- [start:qubit-info]
qubit_info = sq.data.QubitInformation(
    unit="GHz",
    qubit_idx=0,
    anharmonicity=-0.2,
    frequency=5.0,
    drive_strength=0.1,
)
# --8<-- [end:qubit-info]

# --8<-- [start:control]
total_length = 320
pulse = sq.predefined.DragPulseV2(
    duration=total_length,
    qubit_drive_strength=qubit_info.drive_strength,
    dt=2 / 9,
    max_amp=0.5,
    min_theta=0,
    max_theta=2 * jnp.pi,
    min_beta=-5.0,
    max_beta=5.0,
)

control_sequence = sq.control.ControlSequence(
    pulses=[
        pulse,
    ],
    pulse_length_dt=total_length,
)
# --8<-- [end:control]

data_model = sq.predefined.get_predefined_data_model_m1()

# --8<-- [start:gen-syn-dataset]
exp_data, control_seq, _, _ = sq.predefined.generate_experimental_data(
    key=jax.random.key(0),
    hamiltonian=data_model.total_hamiltonian,
    sample_size=1_000,
    strategy=sq.predefined.SimulationStrategy.SHOT,
    get_qubit_information_fn=lambda: data_model.qubit_information,
    get_control_sequence_fn=lambda: data_model.control_sequence,
    method=sq.predefined.WhiteboxStrategy.TROTTER,
    trotter_steps=10_000,
)
# --8<-- [end:gen-syn-dataset]

# --8<-- [start:save-dataset]
from pathlib import Path # noqa: E402

path = Path("./test_data_v1")
# Create the path with parents if not existed already
path.mkdir(parents=True, exist_ok=True)
# Save the experiment with a single liner 😉.
sq.predefined.save_data_to_path(path, exp_data, data_model.control_sequence)
# --8<-- [end:save-dataset]


# --8<-- [start:load-dataset]
loaded_data = sq.predefined.load_data_from_path(
    path,
    hamiltonian_spec=sq.predefined.HamiltonianSpec(
        method=sq.predefined.WhiteboxStrategy.TROTTER,
        trotter_steps=10_000,
    ),
)
# --8<-- [end:load-dataset]