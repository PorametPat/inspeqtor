import jax
from inspeqtor.v2.control import ravel_unravel_fn, ControlSequence
from inspeqtor.experimental.predefined import GaussianPulse, DragPulseV2


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
