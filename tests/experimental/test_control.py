import jax
from jax import numpy as jnp
import inspeqtor.experimental as sq


def test_jax_DRAG_pulse():
    control_sequence = sq.predefined.get_multi_drag_control_sequence_v3()

    key = jax.random.key(0)
    params = control_sequence.sample_params(key)
    total_dt = 80
    t_eval = jnp.linspace(0, total_dt, total_dt + 1)
    waveform = jax.vmap(control_sequence.get_envelope(params))(t_eval)

    # Check that the waveform is of the correct length
    assert waveform.shape == (total_dt + 1,)

    # Check to_dict and from_dict
    control_sequence_dict = control_sequence.to_dict()
    control_sequence_from_dict = sq.control.ControlSequence.from_dict(
        control_sequence_dict, controls=[sq.predefined.MultiDragPulseV3] * 1
    )

    # Check that the waveform is the same after serialization and deserialization
    waveform_from_dict = jax.vmap(control_sequence_from_dict.get_envelope(params))(
        t_eval
    )

    assert isinstance(waveform_from_dict, jnp.ndarray)

    assert jnp.allclose(waveform, waveform_from_dict)


def test_control_sequence_reader(tmp_path):
    control_sequence = sq.predefined.get_multi_drag_control_sequence_v3()
    control_sequence.to_file(tmp_path)

    reader = sq.control.construct_control_sequence_reader(
        controls=[sq.predefined.MultiDragPulseV3]
    )

    reconstruct_control_sequence = reader(tmp_path)

    assert control_sequence == reconstruct_control_sequence
