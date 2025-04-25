import jax.numpy as jnp
import inspeqtor.experimental as sq
from functools import partial


# def gate_optimizer(
#     params,
#     lower,
#     upper,
#     func: typing.Callable,
#     optimizer: optax.GradientTransformation,
#     maxiter: int = 1000,
# ):
#     opt_state = optimizer.init(params)
#     history = []

#     for _ in alive_it(range(maxiter), force_tty=True):
#         grads, aux = jax.grad(func, has_aux=True)(params)
#         updates, opt_state = optimizer.update(grads, opt_state, params)
#         params = optax.apply_updates(params, updates)

#         # Apply projection
#         params = optax.projections.projection_box(params, lower, upper)

#         # Log the history
#         aux["params"] = params
#         history.append(aux)

#     return params, history


# def detune_x_hamiltonian(
#     hamiltonian: typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray],
#     detune: float,
# ) -> typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
#     def detuned_hamiltonian(
#         params: sq.typing.HamiltonianArgs,
#         t: jnp.ndarray,
#         *args,
#         **kwargs,
#     ) -> jnp.ndarray:
#         return hamiltonian(params, t, *args, **kwargs) + detune * sq.constant.X

#     return detuned_hamiltonian


# def get_default_optimizer(n_iterations):
#     return optax.adamw(
#         learning_rate=optax.warmup_cosine_decay_schedule(
#             init_value=1e-6,
#             peak_value=1e-2,
#             warmup_steps=int(0.1 * n_iterations),
#             decay_steps=n_iterations,
#             end_value=1e-6,
#         )
#     )


# def get_gaussian_pulse_sequence(
#     qubit_info: sq.data.QubitInformation,
#     max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
# ):
#     total_length = 320
#     dt = 2 / 9

#     pulse_sequence = sq.pulse.ControlSequence(
#         pulses=[
#             sq.predefined.GaussianPulse(
#                 duration=total_length,
#                 qubit_drive_strength=qubit_info.drive_strength,
#                 dt=dt,
#                 max_amp=max_amp,
#                 min_theta=0.0,
#                 max_theta=2 * jnp.pi,
#             ),
#         ],
#         pulse_length_dt=total_length,
#     )

#     return pulse_sequence


def get_data_model(trotterization_solver: bool = False) -> sq.utils.SyntheticDataModel:
    qubit_info = sq.predefined.get_mock_qubit_information()
    pulse_sequence = sq.predefined.get_gaussian_pulse_sequence(qubit_info=qubit_info)
    dt = 2 / 9

    ideal_hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            get_envelope=sq.predefined.get_envelope_transformer(
                pulse_sequence=pulse_sequence
            ),
            drive_frequency=qubit_info.frequency,
            dt=dt,
        ),
    )

    detune = 0.001
    total_hamiltonian = sq.predefined.detune_x_hamiltonian(
        ideal_hamiltonian, detune * qubit_info.frequency
    )

    solver = sq.predefined.get_single_qubit_whitebox(
        hamiltonian=total_hamiltonian,
        pulse_sequence=pulse_sequence,
        qubit_info=qubit_info,
        dt=dt,
    )

    trotter_solver = sq.physics.make_trotterization_whitebox(
        hamiltonian=total_hamiltonian,
        pulse_sequence=pulse_sequence,
        trotter_steps=1000,
        dt=dt,
    )

    return sq.utils.SyntheticDataModel(
        pulse_sequence=pulse_sequence,
        qubit_information=qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=total_hamiltonian,
        solver=solver if not trotterization_solver else trotter_solver,
        quantum_device=None,
    )


def custom_feature_map(x: jnp.ndarray) -> jnp.ndarray:
    return sq.predefined.polynomial_feature_map(x / (2 * jnp.pi), degree=4)
