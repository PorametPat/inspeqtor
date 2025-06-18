import jax
from jax import numpy as jnp
import inspeqtor.legacy as isq


def test_jax_DRAG_pulse():
    control_sequence = isq.utils.predefined.get_multi_drag_control_sequence_v2()

    key = jax.random.PRNGKey(0)
    params = control_sequence.sample_params(key)
    waveform = control_sequence.get_waveform(params)

    # Check that the waveform is of the correct length
    assert waveform.shape == (control_sequence.pulse_length_dt,)

    # Check to_dict and from_dict
    control_sequence_dict = control_sequence.to_dict()
    control_sequence_from_dict = isq.pulse.JaxBasedPulseSequence.from_dict(
        control_sequence_dict, pulses=[isq.utils.predefined.MultiDragPulseV2] * 4
    )

    # Check that the waveform is the same after serialization and deserialization
    waveform_from_dict = control_sequence_from_dict.get_waveform(params)

    assert isinstance(waveform_from_dict, jnp.ndarray)

    assert jnp.allclose(waveform, waveform_from_dict)


def test_control_sequence_reader(tmp_path):
    control_sequence = isq.utils.predefined.get_multi_drag_control_sequence_v2()
    control_sequence.to_file(str(tmp_path))

    reader = isq.pulse.construct_control_sequence_reader(
        pulses=[isq.utils.predefined.MultiDragPulseV2]
    )

    reconstruct_control_sequence = reader(str(tmp_path))

    assert control_sequence == reconstruct_control_sequence
