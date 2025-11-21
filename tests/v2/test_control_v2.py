from functools import partial
import jax
import jax.numpy as jnp
import inspeqtor as sq
import chex
from flax.traverse_util import flatten_dict


def test_control_sequence():
    duration = 320
    dt = 2 / 9
    pulse = sq.control.library.GaussianPulse(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    drag = sq.control.library.DragPulseV2(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    seq = sq.control.ControlSequence(
        controls={"0": pulse, "1": pulse, "2": drag}, total_dt=duration
    )

    ravel_fn, unravel_fn = sq.control.ravel_unravel_fn(seq.get_structure())

    ctrl_param_sample = seq.sample_params(jax.random.key(0))

    t_eval = jnp.linspace(0, duration, duration + 1)
    waveform = sq.control.sequence_waveform(ctrl_param_sample, t_eval, seq)

    assert ctrl_param_sample == unravel_fn(ravel_fn(ctrl_param_sample))

    re_seq = sq.control.ControlSequence.from_dict(
        seq.to_dict(),
        controls={
            "0": sq.control.library.GaussianPulse,
            "1": sq.control.library.GaussianPulse,
            "2": sq.control.library.DragPulseV2,
        },
    )

    assert seq == re_seq

    r_waveform = sq.control.sequence_waveform(ctrl_param_sample, t_eval, re_seq)

    chex.assert_trees_all_close(waveform, r_waveform)


def test_control_sequence_with_sample_v2():
    duration = 320
    dt = 2 / 9
    pulse = sq.control.library.GaussianPulse(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    drag = sq.control.library.DragPulseV2(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    seq = sq.control.ControlSequence(
        controls={"0": pulse, "1": pulse, "2": drag}, total_dt=duration
    )

    _, _ = sq.control.ravel_unravel_fn(seq.get_structure())

    _ = seq.sample_params_v2(jax.random.key(0))

    # Vectorization test
    jax.vmap(seq.sample_params_v2)(jax.random.split(jax.random.key(0), 10))


def test_ravel_transform():
    qubit_info = sq.data.library.get_mock_qubit_information()
    control_sequence = sq.control.library.get_drag_pulse_v2_sequence(
        qubit_info.drive_strength
    )
    dt = 2 / 9

    total_dt = 320
    t_eval = jnp.linspace(0, total_dt, total_dt + 1)

    key = jax.random.key(0)
    ravel_fn, unravel_fn = sq.control.ravel_unravel_fn(control_sequence.get_structure())
    param = ravel_fn(control_sequence.sample_params(key))

    # Transform at the level of the signal.
    signal_fn = sq.control.ravel_transform(
        sq.physics.make_signal_fn(
            control_sequence.get_envelope, qubit_info.frequency, dt
        ),
        control_sequence,
    )

    signal_v1 = jax.vmap(signal_fn, in_axes=(None, 0))(param, t_eval)

    # Transform at the level of envelope
    signal_fn = sq.physics.make_signal_fn(
        sq.control.ravel_transform(control_sequence.get_envelope, control_sequence),
        qubit_info.frequency,
        dt,
    )

    signal_v2 = jax.vmap(signal_fn, in_axes=(None, 0))(param, t_eval)

    # Check that both are equal
    chex.assert_trees_all_close(signal_v1, signal_v2)

    # Check at the Hamiltonian level

    # Baseline
    hamiltonian_fn = partial(
        sq.physics.library.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.control.ravel_transform(
            sq.physics.make_signal_fn(
                control_sequence.get_envelope, qubit_info.frequency, dt
            ),
            control_sequence,
        ),
    )

    hamiltonian_v1 = jax.vmap(hamiltonian_fn, in_axes=(None, 0))(param, t_eval)

    # Transform at the Hamiltonian level
    hamiltonian_fn = partial(
        sq.physics.library.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.make_signal_fn(
            control_sequence.get_envelope, qubit_info.frequency, dt
        ),
    )

    hamiltonian_fn = sq.control.ravel_transform(hamiltonian_fn, control_sequence)

    hamiltonian_v2 = jax.vmap(hamiltonian_fn, in_axes=(None, 0))(param, t_eval)

    # Check that both are equal
    chex.assert_trees_all_close(hamiltonian_v1, hamiltonian_v2)


def test_arbitrary_pytree_param():
    bounds = {
        "gaussian": {
            "theta": (0, 2 * jnp.pi),
            "nested": {
                "nested_1": (-2, -1),
                "nested_2": (-4, -3),
            },
        }
    }

    lower, upper = sq.control.split_bounds(bounds)

    chex.assert_trees_all_close(
        lower,
        {
            "gaussian": {
                "theta": 0,
                "nested": {
                    "nested_1": -2,
                    "nested_2": -4,
                },
            }
        },
    )

    reconstructed_bounds = sq.control.merge_lower_upper(lower, upper)
    chex.assert_trees_all_close(reconstructed_bounds, bounds)

    chex.assert_trees_all_close(
        upper,
        {
            "gaussian": {
                "theta": 2 * jnp.pi,
                "nested": {
                    "nested_1": -1,
                    "nested_2": -3,
                },
            }
        },
    )

    param = sq.control.nested_sample(jax.random.key(0), bounds)

    assert (
        bounds["gaussian"]["theta"][0]
        <= param["gaussian"]["theta"]
        <= bounds["gaussian"]["theta"][1]
    )
    assert (
        bounds["gaussian"]["nested"]["nested_1"][0]
        <= param["gaussian"]["nested"]["nested_1"]
        <= bounds["gaussian"]["nested"]["nested_1"][1]
    )
    assert (
        bounds["gaussian"]["nested"]["nested_2"][0]
        <= param["gaussian"]["nested"]["nested_2"]
        <= bounds["gaussian"]["nested"]["nested_2"][1]
    )

    assert sq.control.check_bounds(param, bounds)

    # Test vectorization
    params = jax.vmap(sq.control.nested_sample, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 10), bounds
    )

    jax.vmap(sq.control.check_bounds, in_axes=(0, None))(params, bounds)

    # Test unflatten and flatten algorithm
    structure = list(flatten_dict(param).keys())
    ravel_fn, unravel_fn = sq.control.ravel_unravel_fn(structure)

    reconstructed_param = unravel_fn(ravel_fn(param))

    chex.assert_trees_all_close(param, reconstructed_param)

    params_array = jax.vmap(ravel_fn)(params)
    assert params_array.shape == (10, 3)

    reconstructed_params = jax.vmap(unravel_fn)(params_array)

    chex.assert_trees_all_close(params, reconstructed_params)


def test_envelope_fn():
    duration = 320
    dt = 2 / 9
    pulse = sq.control.library.GaussianPulse(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    drag = sq.control.library.DragPulseV2(
        duration=duration, qubit_drive_strength=0.1, dt=dt
    )
    seq = sq.control.ControlSequence(
        controls={"0": pulse, "1": pulse, "2": drag}, total_dt=duration
    )
    t_eval = jnp.linspace(0.0, duration, 321)

    param = seq.sample_params_v2(jax.random.key(0))

    sq.control.get_envelope(param, seq)(0.0)

    jax.vmap(sq.control.get_envelope(param, seq))(t_eval)

    sq.control.envelope_fn(param, jnp.array(0.0), seq)

    r = jax.vmap(sq.control.envelope_fn, in_axes=(None, 0, None))(param, t_eval, seq)
    assert r.shape == (321,)
