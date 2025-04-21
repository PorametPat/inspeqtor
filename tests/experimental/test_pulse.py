import jax
from jax import numpy as jnp
import inspeqtor.experimental as sq


def test_jax_DRAG_pulse():
    pulse_sequence = sq.predefined.get_multi_drag_pulse_sequence_v3()

    key = jax.random.key(0)
    params = pulse_sequence.sample_params(key)
    waveform = pulse_sequence.get_waveform(params)

    # Check that the waveform is of the correct length
    assert waveform.shape == (pulse_sequence.pulse_length_dt,)

    # Check to_dict and from_dict
    pulse_sequence_dict = pulse_sequence.to_dict()
    pulse_sequence_from_dict = sq.pulse.ControlSequence.from_dict(
        pulse_sequence_dict, pulses=[sq.predefined.MultiDragPulseV3] * 1
    )

    # Check that the waveform is the same after serialization and deserialization
    waveform_from_dict = pulse_sequence_from_dict.get_waveform(params)

    assert isinstance(waveform_from_dict, jnp.ndarray)

    assert jnp.allclose(waveform, waveform_from_dict)


def test_pulse_sequence_reader(tmp_path):
    pulse_sequence = sq.predefined.get_multi_drag_pulse_sequence_v3()
    pulse_sequence.to_file(tmp_path)

    reader = sq.pulse.construct_pulse_sequence_reader(
        pulses=[sq.predefined.MultiDragPulseV3]
    )

    reconstruct_pulse_sequence = reader(tmp_path)

    assert pulse_sequence == reconstruct_pulse_sequence
