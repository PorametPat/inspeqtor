import pytest
import jax
import jax.numpy as jnp
from functools import partial

import inspeqtor.experimental as sq
from scipy.stats import unitary_group
import chex
import logging

jax.config.update("jax_enable_x64", True)


def test_signal_func_v3():
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_drag_control_sequence(
        qubit_info.drive_strength
    )
    dt = 2 / 9

    key = jax.random.PRNGKey(0)
    params = control_sequence.sample_params(key)

    signal = sq.physics.signal_func_v3(
        control_sequence.get_envelope,
        qubit_info.frequency,
        dt,
    )

    signal_params = sq.physics.SignalParameters(pulse_params=params, phase=0)

    DURATION = 100

    discreted_signal = signal(
        signal_params, jnp.linspace(0, control_sequence.total_dt * dt, DURATION)
    )

    assert discreted_signal.shape == (DURATION,)


def test_hamiltonian_fn():
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_drag_control_sequence(
        qubit_info.drive_strength
    )
    # control_sequence = sq.predefined.get_gaussian_control_sequence(qubit_info)
    dt = 2 / 9

    key = jax.random.key(0)
    params = control_sequence.sample_params(key)

    total_hamiltonian = sq.physics.gen_hamiltonian_from(
        qubit_informations=[qubit_info],
        coupling_constants=[],
        dims=2,
    )

    drive_terms = list(filter(sq.physics.pick_drive, total_hamiltonian))
    static_terms = list(filter(sq.physics.pick_static, total_hamiltonian))

    signals = {
        drive_term: sq.physics.signal_func_v3(
            control_sequence.get_envelope,
            drive_frequency=qubit_info.frequency,
            dt=dt,
        )
        for drive_term in drive_terms
    }

    signal_param = sq.physics.SignalParameters(pulse_params=params, phase=0)

    hamiltonian_args = {drive_terms[0]: signal_param}

    hamiltonian = partial(
        sq.physics.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    eval_hamiltonian = hamiltonian(hamiltonian_args, jnp.array(1))

    assert eval_hamiltonian.shape == (2, 2)

    eval_hamiltonian = jax.vmap(hamiltonian, in_axes=(None, 0))(
        hamiltonian_args, jnp.array([1, 2])
    )

    assert eval_hamiltonian.shape == (2, 2, 2)


def test_run():
    batch_size = 10
    key = jax.random.PRNGKey(0)
    qubit_info = sq.data.QubitInformation(
        unit="GHz",
        qubit_idx=0,
        anharmonicity=-0.33,
        frequency=5.0,
        drive_strength=0.1,
    )
    dt = 2 / 9

    # Get the pulse sequence
    control_sequence = sq.predefined.get_gaussian_control_sequence(
        qubit_info=qubit_info
    )

    # Sampling the pulse parameters
    # Get the waveforms for each pulse parameters to get the unitaries
    waveforms = []
    for i in range(batch_size):
        key, subkey = jax.random.split(key)
        # pulse_params = control_sequence.sample_params(subkey)
        pulse_params = sq.control.list_of_params_to_array(
            control_sequence.sample_params(subkey),
            control_sequence.get_parameter_names(),
        )
        waveforms.append(pulse_params)

    waveforms = jnp.array(waveforms)

    # Get the simualtor
    # simulator = get_simulator(qubit_info=qubit_info, t_eval=t_eval)
    simulator = sq.predefined.get_single_qubit_rotating_frame_whitebox(
        control_sequence=control_sequence, qubit_info=qubit_info, dt=dt
    )

    # Solve for the unitaries
    # jit the simulator
    jitted_simulator = jax.jit(simulator)
    # batch the simulator
    batched_simulator = jax.vmap(jitted_simulator, in_axes=(0))
    # Get the unitaries
    unitaries = batched_simulator(waveforms)
    # Assert the unitaries shape
    assert unitaries.shape == (batch_size, control_sequence.total_dt, 2, 2)


def setup_crosscheck_setting():
    qubit_info = sq.predefined.get_mock_qubit_information()
    # control_sequence = sq.predefined.get_drag_control_sequence(qubit_info)
    control_sequence = sq.predefined.get_gaussian_control_sequence(qubit_info)

    key = jax.random.PRNGKey(0)
    params = control_sequence.sample_params(key)
    # angle = 0
    # params: list[sq.typing.ParametersDictType] = [
    #     {
    #         "theta": jnp.array(angle),
    #         # "beta": jnp.array(1.0)
    #     }
    # ]

    time_step = 2 / 9
    t_eval = jnp.linspace(
        0,
        control_sequence.total_dt * time_step,
        control_sequence.total_dt,
    )

    return qubit_info, control_sequence, params, t_eval, time_step


def solve_with_manual_rotate():
    qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()
    # NOTE: This hamiltonian is the analytical rotated Hamiltonian of single qubit
    hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v3(
            control_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    jitted_simulator = jax.jit(
        partial(
            sq.physics.solver,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=control_sequence.total_dt * time_step,
        )
    )

    hamil_params = sq.physics.SignalParameters(pulse_params=params, phase=0)

    unitaries_manual_rotated = jitted_simulator(hamil_params)
    return unitaries_manual_rotated


def solve_with_auto_rotate():
    qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()
    # NOTE: For the auto rotating test
    hamiltonian = partial(
        sq.predefined.transmon_hamiltonian,  # This hamiltonian is in the lab frame
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v3(
            control_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    frame = (jnp.pi * qubit_info.frequency) * sq.constant.Z

    rotating_hamiltonian = sq.physics.auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    jitted_simulator = jax.jit(
        partial(
            sq.physics.solver,
            t_eval=t_eval,
            hamiltonian=rotating_hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=control_sequence.total_dt * time_step,
        )
    )

    hamil_params = sq.physics.SignalParameters(pulse_params=params, phase=0)
    auto_rotated_unitaries = jitted_simulator(hamil_params)

    return auto_rotated_unitaries


# def solve_with_pennylane():
#     qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()

#     qml_simulator = get_simulator(
#         qubit_info=qubit_info,
#         t_eval=t_eval,
#         hamiltonian=rotating_transmon_hamiltonian,
#     )
#     qml_unitary = qml_simulator(control_sequence.get_waveform(params))

#     return qml_unitary


def solve_with_auto_hamiltonian_extractor():
    qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()

    # backend = FakeJakartaV2()
    # backend_properties = qk.IBMQDeviceProperties.from_backend(
    #     backend=backend, qubit_indices=[0]
    # )
    # backend_properties.qubit_informations = [qubit_info]

    # coupling_infos = qk.get_coupling_strengths(backend_properties)

    qubit_informations = [sq.predefined.get_mock_qubit_information()]
    coupling_infos = []
    dt = 2 / 9

    total_hamiltonian = sq.physics.gen_hamiltonian_from(
        qubit_informations=qubit_informations,
        coupling_constants=coupling_infos,
        dims=2,
    )

    drive_terms = list(filter(sq.physics.pick_drive, total_hamiltonian))
    static_terms = list(filter(sq.physics.pick_static, total_hamiltonian))

    signals = {
        drive_term: sq.physics.signal_func_v3(
            control_sequence.get_envelope,
            drive_frequency=qubit_informations[int(drive_term[-1])].frequency,
            dt=dt,
        )
        for drive_term in drive_terms
    }

    hamiltonian = partial(
        sq.physics.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    hamiltonian_args = {
        drive_terms[0]: sq.physics.SignalParameters(
            pulse_params=params,
            phase=0.0,
        )
    }

    frame = total_hamiltonian[static_terms[0]].operator
    rotating_hamiltonian = sq.physics.auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    unitaries_hamiltonian_fn = sq.physics.solver(
        hamiltonian_args,
        t_eval=t_eval,
        hamiltonian=rotating_hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
        t0=0,
        t1=control_sequence.total_dt * time_step,
    )

    return unitaries_hamiltonian_fn


def solve_with_signal_v5():
    qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()

    whitebox_v5 = sq.predefined.get_single_qubit_rotating_frame_whitebox(
        control_sequence=control_sequence,
        qubit_info=qubit_info,
        dt=time_step,
    )

    unitaries_v5 = whitebox_v5(
        sq.control.list_of_params_to_array(
            params, control_sequence.get_parameter_names()
        )
    )

    return unitaries_v5


def solver_with_trotterization():
    qubit_info, control_sequence, params, t_eval, time_step = setup_crosscheck_setting()

    hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            get_envelope=sq.predefined.get_envelope_transformer(
                control_sequence=control_sequence
            ),
            drive_frequency=qubit_info.frequency,  # * (1 + detune),
            dt=time_step,
        ),
    )

    whitebox = sq.physics.make_trotterization_solver(
        hamiltonian, control_sequence, time_step, trotter_steps=1000
    )

    unitary = whitebox(
        sq.control.list_of_params_to_array(
            params, control_sequence.get_parameter_names()
        )
    )

    return unitary


@pytest.mark.parametrize(
    "unitaries",
    [
        (solve_with_signal_v5(), solve_with_auto_hamiltonian_extractor()),
        # (solve_with_signal_v5(), solve_with_pennylane()),
        (solve_with_signal_v5(), solve_with_auto_rotate()),
        (solve_with_signal_v5(), solver_with_trotterization()),
    ],
)
def test_crosscheck(unitaries):
    # uni_1, uni_2 = (solve_with_manual_rotate(), solve_with_pennylane())
    uni_1, uni_2 = unitaries
    # fidelities = jax.vmap(sq.physics.gate_fidelity, in_axes=(0, 0))(uni_1, uni_2)
    fidelities = sq.physics.gate_fidelity(uni_1[-1], uni_2[-1])

    # assert jnp.allclose(fidelities, jnp.ones_like(fidelities), rtol=1e-3)
    chex.assert_trees_all_close(fidelities, jnp.ones_like(fidelities))
    # chex.assert_trees_all_close(fidelities[-1], jnp.ones_like(fidelities[-1]))


# There used to be issue with get_drag_control_sequence at near zero area with the diffrax solver.
def test_crosscheck_difflax():
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_drag_control_sequence(
        qubit_info.drive_strength
    )

    key = jax.random.PRNGKey(0)
    params = control_sequence.sample_params(key)

    time_step = 2 / 9
    t_eval = jnp.linspace(
        0, control_sequence.total_dt * (2 / 9), control_sequence.total_dt
    )

    # NOTE: This hamiltonian is the analytical rotated Hamiltonian of single qubit
    hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v3(
            control_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    jitted_simulator = jax.jit(
        partial(
            sq.physics.solver,
            t_eval=t_eval,
            hamiltonian=hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex128),
            t0=0,
            t1=control_sequence.total_dt * time_step,
            rtol=1e-8,
            atol=1e-8,
        )
    )

    hamil_params = sq.physics.SignalParameters(pulse_params=params, phase=0)

    unitaries_manual_rotated = jitted_simulator(hamil_params)

    # NOTE: For the auto rotating test
    hamiltonian = partial(
        sq.predefined.transmon_hamiltonian,  # This hamiltonian is in the lab frame
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v3(
            control_sequence.get_envelope, qubit_info.frequency, 2 / 9
        ),
    )

    frame = (jnp.pi * qubit_info.frequency) * sq.constant.Z

    rotating_hamiltonian = sq.physics.explicit_auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    jitted_simulator = jax.jit(
        partial(
            sq.physics.solver,
            t_eval=t_eval,
            hamiltonian=rotating_hamiltonian,
            y0=jnp.eye(2, dtype=jnp.complex64),
            t0=0,
            t1=control_sequence.total_dt * time_step,
        )
    )
    auto_rotated_unitaries = jitted_simulator(hamil_params)

    qubit_informations = [sq.predefined.get_mock_qubit_information()]
    coupling_infos = []
    dt = 2 / 9

    total_hamiltonian = sq.physics.gen_hamiltonian_from(
        qubit_informations=qubit_informations,
        coupling_constants=coupling_infos,
        dims=2,
    )

    drive_terms = list(filter(sq.physics.pick_drive, total_hamiltonian))
    static_terms = list(filter(sq.physics.pick_static, total_hamiltonian))

    signals = {
        drive_term: sq.physics.signal_func_v3(
            control_sequence.get_envelope,
            drive_frequency=qubit_informations[int(drive_term[-1])].frequency,
            dt=dt,
        )
        for drive_term in drive_terms
    }

    hamiltonian = partial(
        sq.physics.hamiltonian_fn,
        signals=signals,
        hamiltonian_terms=total_hamiltonian,
        static_terms=static_terms,
    )

    hamiltonian_args = {
        drive_terms[0]: sq.physics.SignalParameters(
            pulse_params=params,
            phase=0.0,
        )
    }

    frame = total_hamiltonian[static_terms[0]].operator
    rotating_hamiltonian = sq.physics.explicit_auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )

    unitaries_hamiltonian_fn = sq.physics.solver(
        hamiltonian_args,
        t_eval=t_eval,
        hamiltonian=rotating_hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
        t0=0,
        t1=control_sequence.total_dt * time_step,
    )

    whitebox_v5 = sq.predefined.get_single_qubit_rotating_frame_whitebox(
        control_sequence=control_sequence,
        qubit_info=qubit_info,
        dt=2 / 9,
    )

    unitaries_v5 = whitebox_v5(
        sq.control.list_of_params_to_array(
            params, control_sequence.get_parameter_names()
        )
    )

    unitaries_tuple = [
        (
            unitaries_manual_rotated,
            unitaries_hamiltonian_fn,
            "From automatic hamiltonian construction fn",
        ),
        (unitaries_manual_rotated, unitaries_v5, "Predefined auto rotate"),
        (unitaries_manual_rotated, auto_rotated_unitaries, "Auto rotate"),
    ]

    for uni_1, uni_2, name in unitaries_tuple:
        logging.info(f"Test with: {name}")
        fidelities = jax.vmap(sq.physics.gate_fidelity, in_axes=(0, 0))(uni_1, uni_2)

        chex.assert_trees_all_close(fidelities, jnp.ones_like(fidelities))


def get_diffrax_solver(hamiltonian, control_seq, dt):
    t_eval = jnp.linspace(
        0, control_seq.total_dt * dt, control_seq.total_dt
    )
    diffrax_solver = partial(
        sq.physics.solver,
        t_eval=t_eval,
        hamiltonian=hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex128),
        t0=0,
        t1=control_seq.total_dt * dt,
        rtol=1e-7,
        atol=1e-7,
    )
    return diffrax_solver


def get_trotter_solver(hamiltonian, control_seq, dt):
    TROTTER_STEPS = 10_000
    trotter_solver = sq.physics.make_trotterization_solver(
        hamiltonian=hamiltonian,
        control_sequence=control_seq,
        dt=dt,
        trotter_steps=TROTTER_STEPS,
    )
    return trotter_solver


def test_trotter_diffrax_close():
    data_model = sq.predefined.get_predefined_data_model_m1()

    hamiltonian = data_model.total_hamiltonian
    control_seq = data_model.control_sequence
    dt = data_model.dt

    trotter_solver = get_trotter_solver(hamiltonian, control_seq, dt)

    diffrax_solver = get_diffrax_solver(hamiltonian, control_seq, dt)

    _, l2a_fn = sq.control.get_param_array_converter(control_seq)
    params = l2a_fn(control_seq.sample_params(jax.random.key(0)))

    trotter_unitary = trotter_solver(params)
    diffrax_unitary = diffrax_solver(params)

    assert sq.physics.gate_fidelity(trotter_unitary[-1], diffrax_unitary[-1]) > 0.999


def get_manual_rotated_hamiltonian(qubit_info, control_seq, dt):
    hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            sq.control.get_envelope_transformer(control_seq), qubit_info.frequency, dt
        ),
    )
    return hamiltonian


def get_auto_rotated_hamiltonian(qubit_info, control_seq, dt):
    hamiltonian = partial(
        sq.predefined.transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            sq.control.get_envelope_transformer(control_seq), qubit_info.frequency, dt
        ),
    )
    frame = (jnp.pi * qubit_info.frequency) * sq.constant.Z

    rotating_hamiltonian = sq.physics.auto_rotating_frame_hamiltonian(
        hamiltonian, frame
    )
    return rotating_hamiltonian


def test_auto_rotate_trotter():
    data_model = sq.predefined.get_predefined_data_model_m1()

    qubit_info = sq.predefined.get_mock_qubit_information()
    control_seq = data_model.control_sequence
    dt = data_model.dt
    _, l2a_fn = sq.control.get_param_array_converter(control_seq)
    params = l2a_fn(control_seq.sample_params(jax.random.key(0)))

    hamiltonian_1 = get_manual_rotated_hamiltonian(qubit_info, control_seq, dt)
    trotter_solver = get_trotter_solver(hamiltonian_1, control_seq, dt)
    unitary_1 = trotter_solver(params)

    hamiltonian_2 = get_auto_rotated_hamiltonian(qubit_info, control_seq, dt)
    trotter_solver = get_trotter_solver(hamiltonian_2, control_seq, dt)
    unitary_2 = trotter_solver(params)
    fideilities = jax.vmap(sq.physics.gate_fidelity, in_axes=(0, 0))(
        unitary_1, unitary_2
    )
    chex.assert_trees_all_close(fideilities, jnp.ones_like(fideilities))


def test_auto_rotate_diffrax():
    data_model = sq.predefined.get_predefined_data_model_m1()

    qubit_info = sq.predefined.get_mock_qubit_information()
    control_seq = data_model.control_sequence
    dt = data_model.dt
    _, l2a_fn = sq.control.get_param_array_converter(control_seq)
    params = l2a_fn(control_seq.sample_params(jax.random.key(0)))

    hamiltonian_1 = get_manual_rotated_hamiltonian(qubit_info, control_seq, dt)
    trotter_solver = get_diffrax_solver(hamiltonian_1, control_seq, dt)
    unitary_1 = trotter_solver(params)

    hamiltonian_2 = get_auto_rotated_hamiltonian(qubit_info, control_seq, dt)
    trotter_solver = get_diffrax_solver(hamiltonian_2, control_seq, dt)
    unitary_2 = trotter_solver(params)
    fideilities = jax.vmap(sq.physics.gate_fidelity, in_axes=(0, 0))(
        unitary_1, unitary_2
    )
    chex.assert_trees_all_close(fideilities, jnp.ones_like(fideilities))


@pytest.mark.parametrize(
    "state",
    [
        jnp.array([[1, 0], [0, 0]]),
        jnp.array([[0, 0], [0, 1]]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        jnp.array([[0.5, -0.5j], [0.5j, 0.5]]),
    ],
)
def test_state_tomography(state):
    exp_X = jnp.trace(state @ sq.constant.X)
    exp_Y = jnp.trace(state @ sq.constant.Y)
    exp_Z = jnp.trace(state @ sq.constant.Z)

    reconstructed_state = sq.physics.state_tomography(exp_X, exp_Y, exp_Z)

    assert jnp.allclose(state, reconstructed_state)

    # Check valid density matrix
    sq.physics.check_valid_density_matrix(reconstructed_state)


def wrong_to_superop(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(U, U.conj())


def test_SX():
    state_0 = sq.data.State.from_label("0", dm=True)
    state_1 = sq.data.State.from_label("1", dm=True)
    SX = sq.constant.SX

    state_f = SX @ SX @ state_0 @ SX.conj().T @ SX.conj().T

    assert jnp.allclose(state_1, state_f)


def theoretical_fidelity_of_amplitude_dampling_channel_with_itself(gamma):
    process_fidelity = 1 - gamma + (gamma**2) / 2

    return (2 * process_fidelity + 1) / (2 + 1)


def test_direct_AFG_estimation_coefficients():
    # The order is [XX(+1), XX(-1), XY(+1), XY(-1), ...]
    expected_sx_coefficients = jnp.array(
        [
            1,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -1,
            1,
            0,
            0,
            1,
            -1,
            0,
            0,
        ]
    )

    coefficients = sq.physics.direct_AFG_estimation_coefficients(sq.constant.SX)

    assert jnp.allclose(coefficients, expected_sx_coefficients)


def test_direct_AFG_estimation():
    target_unitary = jnp.array(unitary_group.rvs(2))

    expvals = []
    for exp in sq.constant.default_expectation_values_order:
        expval = sq.utils.calculate_exp(
            target_unitary, exp.observable_matrix, exp.initial_density_matrix
        )
        expvals.append(expval)

    coefficients = sq.physics.direct_AFG_estimation_coefficients(target_unitary)

    AFG = sq.physics.direct_AFG_estimation(coefficients, jnp.array(expvals))

    assert jnp.allclose(AFG, 1)
