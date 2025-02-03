import jax
import jax.numpy as jnp
import optax
from alive_progress import alive_it
import typing
import inspeqtor.experimental as sq
from dataclasses import dataclass


def gate_optimizer(
    params,
    lower,
    upper,
    func: typing.Callable,
    optimizer: optax.GradientTransformation,
    maxiter: int = 1000,
):
    opt_state = optimizer.init(params)
    history = []

    for _ in alive_it(range(maxiter), force_tty=True):
        grads, aux = jax.grad(func, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Apply projection
        params = optax.projections.projection_box(params, lower, upper)

        # Log the history
        aux["params"] = params
        history.append(aux)

    return params, history


def detune_x_hamiltonian(
    hamiltonian: typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    detune: float,
) -> typing.Callable[[sq.typing.HamiltonianArgs, jnp.ndarray], jnp.ndarray]:
    def detuned_hamiltonian(
        params: sq.typing.HamiltonianArgs,
        t: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return hamiltonian(params, t, *args, **kwargs) + detune * sq.constant.X

    return detuned_hamiltonian


def get_default_optimizer(n_iterations):
    return optax.adamw(
        learning_rate=optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-2,
            warmup_steps=int(0.1 * n_iterations),
            decay_steps=n_iterations,
            end_value=1e-6,
        )
    )


def gaussian_envelope(amp, center, sigma):
    def g_fn(t):
        return (amp / (jnp.sqrt(2 * jnp.pi) * sigma)) * jnp.exp(
            -((t - center) ** 2) / (2 * sigma**2)
        )

    return g_fn


@dataclass
class GaussianPulse(sq.pulse.BasePulse):
    duration: int
    # beta: float
    qubit_drive_strength: float
    dt: float
    max_amp: float = 0.25

    min_theta: float = 0.0
    max_theta: float = 2 * jnp.pi

    def __post_init__(self):
        self.t_eval = jnp.arange(self.duration)

        # This is the correction factor that will cancel the factor in the front of hamiltonian
        self.correction = 2 * jnp.pi * self.qubit_drive_strength * self.dt

        # The standard derivation of Gaussian pulse is keep fixed for the given max_amp
        self.sigma = jnp.sqrt(2 * jnp.pi) / (self.max_amp * self.correction)

        # The center position is set at the center of the duration
        self.center_position = self.duration // 2

    def get_bounds(
        self,
    ) -> tuple[sq.typing.ParametersDictType, sq.typing.ParametersDictType]:
        lower = {}
        upper = {}

        lower["theta"] = self.min_theta
        upper["theta"] = self.max_theta

        return lower, upper

    def get_envelope(
        self, params: sq.typing.ParametersDictType
    ) -> typing.Callable[..., typing.Any]:
        # The area of Gaussian to be rotate to,
        area = (
            params["theta"] / self.correction
        )  # NOTE: Choice of area is arbitrary e.g. pi pulse

        return gaussian_envelope(
            amp=area, center=self.center_position, sigma=self.sigma
        )


def get_gaussian_pulse_sequence(
    qubit_info: sq.data.QubitInformation,
    max_amp: float = 0.5,  # NOTE: Choice of maximum amplitude is arbitrary
):
    total_length = 320
    dt = 2 / 9

    pulse_sequence = sq.pulse.PulseSequence(
        pulses=[
            GaussianPulse(
                duration=total_length,
                qubit_drive_strength=qubit_info.drive_strength,
                dt=dt,
                max_amp=max_amp,
                min_theta=0.0,
                max_theta=2 * jnp.pi,
            ),
        ],
        pulse_length_dt=total_length,
    )

    return pulse_sequence
