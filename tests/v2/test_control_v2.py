from functools import partial
import jax
import jax.numpy as jnp
import inspeqtor as sq
from inspeqtor.v2.control import (
    ravel_unravel_fn,
    ControlSequence,
    sequence_waveform,
)
from inspeqtor.experimental.predefined import GaussianPulse, DragPulseV2
import chex


def test_control_sequence():
    duration = 320
    dt = 2 / 9
    pulse = GaussianPulse(duration=duration, qubit_drive_strength=0.1, dt=dt)
    drag = DragPulseV2(duration=duration, qubit_drive_strength=0.1, dt=dt)
    seq = ControlSequence(
        controls={"0": pulse, "1": pulse, "2": drag}, total_dt=duration
    )

    ravel_fn, unravel_fn = ravel_unravel_fn(seq)

    ctrl_param_sample = seq.sample_params(jax.random.key(0))

    t_eval = jnp.linspace(0, duration, duration + 1)
    waveform = sequence_waveform(ctrl_param_sample, t_eval, seq)

    assert ctrl_param_sample == unravel_fn(ravel_fn(ctrl_param_sample))

    re_seq = ControlSequence.from_dict(
        seq.to_dict(),
        controls={
            "0": GaussianPulse,
            "1": GaussianPulse,
            "2": DragPulseV2,
        },
    )

    assert seq == re_seq

    r_waveform = sequence_waveform(ctrl_param_sample, t_eval, re_seq)

    chex.assert_trees_all_close(waveform, r_waveform)


def test_ravel_transform():
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_drag_pulse_v2_sequence(
        qubit_info.drive_strength
    )
    dt = 2 / 9

    total_dt = 320
    t_eval = jnp.linspace(0, total_dt, total_dt + 1)

    key = jax.random.key(0)
    ravel_fn, unravel_fn = ravel_unravel_fn(control_sequence)
    param = ravel_fn(control_sequence.sample_params(key))

    # Transform at the level of the signal.
    signal_fn = sq.control.ravel_transform(
        sq.physics.signal_func_v5(
            control_sequence.get_envelope, qubit_info.frequency, dt
        ),
        control_sequence,
    )

    signal_v1 = jax.vmap(signal_fn, in_axes=(None, 0))(param, t_eval)

    # Transform at the level of envelope
    signal_fn = sq.physics.signal_func_v5(
        sq.control.get_envelope_transformer(control_sequence), qubit_info.frequency, dt
    )

    signal_v2 = jax.vmap(signal_fn, in_axes=(None, 0))(param, t_eval)

    # Check that both are equal
    chex.assert_trees_all_close(signal_v1, signal_v2)

    # Check at the Hamiltonian level

    # Baseline
    hamiltonian_fn = partial(
        sq.experimental.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.control.ravel_transform(
            sq.physics.signal_func_v5(
                control_sequence.get_envelope, qubit_info.frequency, dt
            ),
            control_sequence,
        ),
    )

    hamiltonian_v1 = jax.vmap(hamiltonian_fn, in_axes=(None, 0))(param, t_eval)

    # Transform at the Hamiltonian level
    hamiltonian_fn = partial(
        sq.experimental.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            control_sequence.get_envelope, qubit_info.frequency, dt
        ),
    )

    hamiltonian_fn = sq.control.ravel_transform(hamiltonian_fn, control_sequence)

    hamiltonian_v2 = jax.vmap(hamiltonian_fn, in_axes=(None, 0))(param, t_eval)

    # Check that both are equal
    chex.assert_trees_all_close(hamiltonian_v1, hamiltonian_v2)
