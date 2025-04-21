from flax import struct
import jax
import jax.numpy as jnp
import typing
from dataclasses import dataclass
from enum import Enum
import diffrax  # type: ignore
from .typing import ParametersDictType
from .data import QubitInformation
from .constant import X, Y, Z
from .pulse import ControlSequence


@struct.dataclass
class SignalParameters:
    pulse_params: list[ParametersDictType]
    phase: float
    random_key: None | jnp.ndarray = None


class SimulatorFnV2(typing.Protocol):
    def __call__(
        self,
        args: SignalParameters,
    ) -> jnp.ndarray: ...


class TermType(Enum):
    STATIC = 0
    ANHAMONIC = 1
    DRIVE = 2
    CONTROL = 3
    COUPLING = 4


@struct.dataclass
class HamiltonianTerm:
    qubit_idx: int | tuple[int, int]
    type: TermType
    controlable: bool
    operator: jnp.ndarray


@dataclass
class CouplingInformation:
    qubit_indices: tuple[int, int]
    coupling_strength: float


HamiltonianArgs = typing.TypeVar("HamiltonianArgs")


@dataclass
class ChannelID:
    qubit_idx: int | tuple[int, int]
    type: TermType

    def hash(self):
        if self.type == TermType.COUPLING:
            assert isinstance(self.qubit_idx, tuple)
            return f"ct/{self.type.value}/{self.qubit_idx[0]}/{self.qubit_idx[1]}"
        return f"ct/{self.type.value}/{self.qubit_idx}"

    @classmethod
    def from_hash(cls, hash: str):
        parts = hash.split("/")
        if parts[1] == "4":
            return cls(
                qubit_idx=(int(parts[2]), int(parts[3])), type=TermType(int(parts[1]))
            )
        return cls(qubit_idx=int(parts[2]), type=TermType(int(parts[1])))


def normalizer(matrix: jnp.ndarray) -> jnp.ndarray:
    """Normalize the given matrix with QR decomposition and return matrix Q
       which is unitary

    Args:
        matrix (jnp.ndarray): The matrix to normalize to unitary matrix

    Returns:
        jnp.ndarray: The unitary matrix
    """
    return jnp.linalg.qr(matrix).Q  # type: ignore


def solver(
    args: HamiltonianArgs,
    t_eval: jnp.ndarray,
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    y0: jnp.ndarray,
    t0: float,
    t1: float,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    max_steps: int = int(2**16),
) -> jnp.ndarray:
    """Solve the Schrodinger equation using the given Hamiltonian

    Args:
        args (HamiltonianArgs): The arguments for the Hamiltonian
        t_eval (jnp.ndarray): The time points to evaluate the solution
        hamiltonian (typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray]): The Hamiltonian function
        y0 (jnp.ndarray): The initial state, set to jnp.eye(2, dtype=jnp.complex128) for unitary matrix
        t0 (float): The initial time
        t1 (float): The final time
        rtol (float, optional): _description_. Defaults to 1e-7.
        atol (float, optional): _description_. Defaults to 1e-7.
        max_steps (int, optional): The maxmimum step of evalution of solver. Defaults to int(2**16).

    Returns:
        jnp.ndarray: The solution of the Schrodinger equation at the given time points
    """

    # * Increase time_step to increase accuracy of solver,
    # *     then you have to increase the max_steps too.
    # * Using just a basic solver
    def rhs(t: jnp.ndarray, y: jnp.ndarray, args: HamiltonianArgs):
        return -1j * hamiltonian(args, t) @ y

    term = diffrax.ODETerm(rhs)  # type: ignore
    solver = diffrax.Tsit5()

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        stepsize_controller=diffrax.PIDController(
            rtol=rtol,
            atol=atol,
        ),
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t_eval),
        max_steps=max_steps,
    )

    # Normailized the solution
    ys = solution.ys
    assert isinstance(ys, jnp.ndarray)

    return jax.vmap(normalizer)(ys)


def auto_rotating_frame_hamiltonian(
    hamiltonian: typing.Callable[[HamiltonianArgs, jnp.ndarray], jnp.ndarray],
    frame: jnp.ndarray,
    explicit_deriv: bool = False,
):
    """Implement the Hamiltonian in the rotating frame with
    H_I = U(t) @ H @ U^dagger(t) + i * U(t) @ dU^dagger(t)/dt

    Args:
        hamiltonian (Callable): The hamiltonian function
        frame (jnp.ndarray): The frame matrix
    """

    is_diagonal = False
    # Check if the frame is diagonal matrix
    if jnp.count_nonzero(frame - jnp.diag(jnp.diagonal(frame))) == 0:
        is_diagonal = True

    # Check if the jax_enable_x64 is True
    if not jax.config.read("jax_enable_x64") or is_diagonal:

        def frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
            # NOTE: This is the same as the below, as we sure that the frame is diagonal
            return jnp.diag(jnp.exp(1j * jnp.diagonal(frame) * t))

    else:

        def frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
            return jax.scipy.linalg.expm(1j * frame * t)

    def derivative_frame_unitary(t: jnp.ndarray) -> jnp.ndarray:
        # NOTE: Assume that the frame is time independent.
        return 1j * frame @ frame_unitary(t)

    def rotating_frame_hamiltonian_v0(args: HamiltonianArgs, t: jnp.ndarray):
        return frame_unitary(t) @ hamiltonian(args, t) @ jnp.transpose(
            jnp.conjugate(frame_unitary(t))
        ) + 1j * (
            derivative_frame_unitary(t) @ jnp.transpose(jnp.conjugate(frame_unitary(t)))
        )

    def rotating_frame_hamiltonian(args: HamiltonianArgs, t: jnp.ndarray):
        # NOTE: Assume that the product of derivative and conjugate of frame unitary is identity
        return (
            frame_unitary(t)
            @ hamiltonian(args, t)
            @ jnp.transpose(jnp.conjugate(frame_unitary(t)))
            - frame
        )

    return (
        rotating_frame_hamiltonian_v0 if explicit_deriv else rotating_frame_hamiltonian
    )


def a(dims: int) -> jnp.ndarray:
    """Annihilation operator of given dims

    Args:
        dims (int): Number of states

    Returns:
        jnp.ndarray: Annihilation operator
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dims)), 1)


def a_dag(dims: int) -> jnp.ndarray:
    """Creation operator of given dims

    Args:
        dims (int): Number of states

    Returns:
        jnp.ndarray: Creation operator
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dims)), -1)


def N(dims: int) -> jnp.ndarray:
    """Number operator of given dims

    Args:
        dims (int): Number of states

    Returns:
        jnp.ndarray: Number operator
    """
    return jnp.diag(jnp.arange(dims))


def tensor(operator: jnp.ndarray, position: int, total_qubits: int):
    return jnp.kron(
        jnp.eye(operator.shape[0] ** position),
        jnp.kron(operator, jnp.eye(operator.shape[0] ** (total_qubits - position - 1))),
    )


def gen_hamiltonian_from(
    qubit_informations: list[QubitInformation],
    coupling_constants: list[CouplingInformation],
    dims: int = 2,
) -> dict[str, HamiltonianTerm]:
    """Generate dict of Hamiltonian from given qubits and coupling information.

    Args:
        qubit_informations (list[QubitInformation]): Qubit information
        coupling_constants (list[CouplingInformation]): Coupling information
        dims (int, optional): The level of the quantum system. Defaults to 2, i.e. qubit system.

    Returns:
        dict[str, HamiltonianTerm]: _description_
    """
    num_qubits = len(qubit_informations)

    operators: dict[str, HamiltonianTerm] = {}

    for idx, qubit in enumerate(qubit_informations):
        # The static Hamiltonian terms
        static_i = 2 * jnp.pi * qubit.frequency * (jnp.eye(dims) - 2 * N(dims)) / 2

        operators[ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.STATIC).hash()] = (
            HamiltonianTerm(
                qubit_idx=qubit.qubit_idx,
                type=TermType.STATIC,
                controlable=False,
                operator=tensor(static_i, idx, num_qubits),
            )
        )

        # The anharmonicity term
        anhar_i = 2 * jnp.pi * qubit.anharmonicity * (N(dims) @ N(dims) - N(dims)) / 2

        operators[
            ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.ANHAMONIC).hash()
        ] = HamiltonianTerm(
            qubit_idx=qubit.qubit_idx,
            type=TermType.ANHAMONIC,
            controlable=False,
            operator=tensor(anhar_i, idx, num_qubits),
        )

        # The drive terms
        drive_i = 2 * jnp.pi * qubit.drive_strength * (a(dims) + a_dag(dims))

        operators[ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.DRIVE).hash()] = (
            HamiltonianTerm(
                qubit_idx=qubit.qubit_idx,
                type=TermType.DRIVE,
                controlable=True,
                operator=tensor(drive_i, idx, num_qubits),
            )
        )

        # The control terms that drive with another qubit frequency
        control_i = 2 * jnp.pi * qubit.drive_strength * (a(dims) + a_dag(dims))

        operators[
            ChannelID(qubit_idx=qubit.qubit_idx, type=TermType.CONTROL).hash()
        ] = HamiltonianTerm(
            qubit_idx=qubit.qubit_idx,
            type=TermType.CONTROL,
            controlable=True,
            operator=tensor(control_i, idx, num_qubits),
        )

    for coupling in coupling_constants:
        # Add the coupling constant to the Hamiltonian
        c_1 = tensor(a(dims), coupling.qubit_indices[0], num_qubits) @ tensor(
            a_dag(dims), coupling.qubit_indices[1], num_qubits
        )
        c_2 = tensor(a_dag(dims), coupling.qubit_indices[0], num_qubits) @ tensor(
            a(dims), coupling.qubit_indices[1], num_qubits
        )
        coupling_ij = 2 * jnp.pi * coupling.coupling_strength * (c_1 + c_2)

        operators[
            ChannelID(
                qubit_idx=(coupling.qubit_indices[0], coupling.qubit_indices[1]),
                type=TermType.COUPLING,
            ).hash()
        ] = HamiltonianTerm(
            qubit_idx=(coupling.qubit_indices[0], coupling.qubit_indices[1]),
            type=TermType.COUPLING,
            controlable=False,
            operator=coupling_ij,
        )

    return operators


def hamiltonian_fn(
    args: dict[str, SignalParameters],
    t: jnp.ndarray,
    signals: dict[str, typing.Callable[[SignalParameters, jnp.ndarray], jnp.ndarray]],
    hamiltonian_terms: dict[str, HamiltonianTerm],
    static_terms: list[str],
) -> jnp.ndarray:
    """Hamiltonian function to be used whitebox.
    Expect to be used in partial form, i.e. making `signals`, `hamiltonian_terms`, and `static_terms` static arguments.

    Args:
        args (dict[str, SignalParameters]): Control parameter
        t (jnp.ndarray): Time to evaluate
        signals (dict[str, typing.Callable[[SignalParameters, jnp.ndarray], jnp.ndarray]]): Signal function of the control
        hamiltonian_terms (dict[str, HamiltonianTerm]): Dict of Hamiltonian terms, where key is channel
        static_terms (list[str]): list of channel id specifing the static term.

    Returns:
        jnp.ndarray: _description_
    """
    # Match the args with signal
    drives = jnp.array(
        [
            signal(args[channel_id], t) * hamiltonian_terms[channel_id].operator
            for channel_id, signal in signals.items()
        ]
    )

    statics = jnp.array(
        [hamiltonian_terms[static_term].operator for static_term in static_terms]
    )

    return jnp.sum(drives, axis=0) + jnp.sum(statics, axis=0)


def pick_non_controlable(channel: str):
    return int(channel[3]) not in [TermType.DRIVE.value, TermType.CONTROL.value]


def pick_drive(channel: str):
    return int(channel[3]) == TermType.DRIVE.value


def pick_static(channel: str):
    return int(channel[3]) == TermType.STATIC.value


def signal_func_v3(get_envelope: typing.Callable, drive_frequency: float, dt: float):
    """Make the envelope function into signal with drive frequency

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    def signal(pulse_parameters: SignalParameters, t: jnp.ndarray):
        return jnp.real(
            get_envelope(pulse_parameters.pulse_params)(t / dt)
            * jnp.exp(
                1j * ((2 * jnp.pi * drive_frequency * t) + pulse_parameters.phase)
            )
        )

    return signal


ControlParam = typing.TypeVar("ControlParam")


def signal_func_v5(
    get_envelope: typing.Callable[
        [ControlParam], typing.Callable[[jnp.ndarray], jnp.ndarray]
    ],
    drive_frequency: float,
    dt: float,
):
    """Make the envelope function into signal with drive frequency

    Args:
        get_envelope (Callable): The envelope function in unit of dt
        drive_frequency (float): drive freuqency in unit of GHz
        dt (float): The dt provived will be used to convert envelope unit to ns,
                    set to 1 if the envelope function is already in unit of ns
    """

    def signal(pulse_parameters: ControlParam, t: jnp.ndarray):
        return jnp.real(
            get_envelope(pulse_parameters)(t / dt)
            * jnp.exp(1j * (2 * jnp.pi * drive_frequency * t))
        )

    return signal


def gate_fidelity(U: jnp.ndarray, V: jnp.ndarray) -> jnp.ndarray:
    """Calculate the gate fidelity between U and V

    Args:
        U (jnp.ndarray): Unitary operator to be targetted
        V (jnp.ndarray): Unitary operator to be compared

    Returns:
        jnp.ndarray: Gate fidelity
    """
    up = jnp.trace(U.conj().T @ V)
    down = jnp.sqrt(jnp.trace(U.conj().T @ U) * jnp.trace(V.conj().T @ V))

    return jnp.abs(up / down) ** 2


def process_fidelity(superop: jnp.ndarray, target_superop: jnp.ndarray) -> jnp.ndarray:
    # Calculate the fidelity
    # TODO: check if this is correct
    fidelity = jnp.trace(jnp.matmul(target_superop.conj().T, superop)) / 4
    return fidelity


def avg_gate_fidelity_from_superop(
    U: jnp.ndarray, target_U: jnp.ndarray
) -> jnp.ndarray:
    dim = 2
    avg_fid = (dim * process_fidelity(U, target_U) + 1) / (dim + 1)
    return jnp.real(avg_fid)


def to_superop(U: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(U.conj(), U)


def state_tomography(exp_X: jnp.ndarray, exp_Y: jnp.ndarray, exp_Z: jnp.ndarray):
    return (jnp.eye(2) + exp_X * X + exp_Y * Y + exp_Z * Z) / 2


def check_valid_density_matrix(rho: jnp.ndarray):
    """Check if the provided matrix is valid density matrix

    Args:
        rho (jnp.ndarray): _description_
    """
    # Check if the density matrix is valid
    assert jnp.allclose(jnp.trace(rho), 1.0), "Density matrix is not trace 1"
    assert jnp.allclose(rho, rho.conj().T), "Density matrix is not Hermitian"


def check_hermitian(op: jnp.ndarray):
    """Check if the provided matrix is Hermitian

    Args:
        op (jnp.ndarray): Matrix to be assert
    """
    assert jnp.allclose(op, op.conj().T), "Matrix is not Hermitian"


def direct_AFG_estimation(
    coefficients: jnp.ndarray,
    expectation_values: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate single qubit average gate fidelity from expectation value
    This function should be used with `direct_AFG_estimation_coefficients`

    >>> coefficients = direct_AFG_estimation_coefficients(unitary)
    ... agf = direct_AFG_estimation(coefficients, expectation_value)

    Args:
        coefficients (jnp.ndarray): The coefficients return from `direct_AFG_estimation_coefficients`
        expectation_values (jnp.ndarray): The expectation values assume to be shape of (..., 18) with order of `sq.constant.default_expectation_values_order`

    Returns:
        jnp.ndarray: Average Gate Fidelity
    """
    return (1 / 2) + ((1 / 12) * jnp.dot(coefficients, expectation_values))


def direct_AFG_estimation_coefficients(target_unitary: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected coefficients to be used for AGF calculation using `direct_AFG_estimation`.
    The order of coefficients is the same as `sq.constant.default_expectation_values_order`

    Args:
        target_unitary (jnp.ndarray): Target unitary to be computed for coefficient

    Returns:
        jnp.ndarray: Coefficients for AGF calculation.
    """
    coefficients = []
    for pauli_i in [X, Y, Z]:
        for pauli_j in [X, Y, Z]:
            pauli_coeff = (1 / 2) * jnp.trace(
                pauli_i @ target_unitary @ pauli_j @ target_unitary.conj().T
            )
            for state_coeff in [1, -1]:
                coeff = state_coeff * pauli_coeff
                coefficients.append(coeff)

    return jnp.real(jnp.array(coefficients))


def calculate_exp(
    unitary: jnp.ndarray, operator: jnp.ndarray, density_matrix: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the expectation value for given unitary, observable (operator), initial state (density_matrix).
    Shape of all arguments must be boardcastable.

    Args:
        unitary (jnp.ndarray): Unitary operator
        operator (jnp.ndarray): Quantum Observable
        density_matrix (jnp.ndarray): Intial state in form of density matrix.

    Returns:
        jnp.ndarray: Expectation value of quantum observable.
    """
    rho = jnp.matmul(
        unitary, jnp.matmul(density_matrix, unitary.conj().swapaxes(-2, -1))
    )
    temp = jnp.matmul(rho, operator)
    return jnp.real(jnp.sum(jnp.diagonal(temp, axis1=-2, axis2=-1), axis=-1))


def unitaries_prod(
    prev_unitary: jnp.ndarray, curr_unitary: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Function to be used for trotterization Whitebox

    Args:
        prev_unitary (jnp.ndarray): Product of cummulate Unitary operator.
        curr_unitary (jnp.ndarray): The next Unitary operator to be multiply.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Product of previous unitart and current unitary.
    """
    prod_unitary = prev_unitary @ curr_unitary
    return prod_unitary, prod_unitary


def make_trotterization_whitebox(
    hamiltonian: typing.Callable[..., jnp.ndarray],
    pulse_sequence: ControlSequence,
    dt: float = 2 / 9,
    trotter_steps: int = 1000,
):
    """Retutn whitebox function compute using Trotterization strategy.

    Args:
        hamiltonian (typing.Callable[..., jnp.ndarray]): The Hamiltonian function of the system
        pulse_sequence (PulseSequence): The pulse sequence instance
        dt (float, optional): The duration of time step in nanosecond. Defaults to 2/9.
        trotter_steps (int, optional): The number of trotterization step. Defaults to 1000.

    Returns:
        typing.Callable[..., jnp.ndarray]: Trotterization Whitebox function
    """
    hamiltonian = jax.jit(hamiltonian)
    time_step = jnp.linspace(0, pulse_sequence.pulse_length_dt * dt, trotter_steps)

    def whitebox(pulse_parameter: jnp.ndarray):
        hamiltonians = jax.vmap(hamiltonian, in_axes=(None, 0))(
            pulse_parameter, time_step
        )
        unitaries = jax.scipy.linalg.expm(
            -1j * (time_step[1] - time_step[0]) * hamiltonians
        )
        # * Nice explanation of scan
        # * https://www.nelsontang.com/blog/a-friendly-introduction-to-scan-with-jax
        _, unitaries = jax.lax.scan(
            unitaries_prod, jnp.eye(2, dtype=jnp.complex128), unitaries
        )
        return unitaries

    return whitebox
