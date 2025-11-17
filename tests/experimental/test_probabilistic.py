import pytest
import jax
import jax.numpy as jnp
import numpyro.distributions as dist  # type: ignore
from inspeqtor.experimental.probabilistic import (
    binary_to_eigenvalue,
    batched_matmul,
    get_trace,
    dense_layer,
)
from inspeqtor.experimental.utils import (
    eigenvalue_to_binary,
    expectation_value_to_eigenvalue,
)
import inspeqtor.experimental as sq
import chex
import typing
import numpyro
from functools import partial
from numpyro.contrib.module import random_flax_module, random_nnx_module

from numpyro.infer import SVI, TraceMeanField_ELBO
from flax import nnx

jax.config.update("jax_enable_x64", True)


def test_eigenvalue_binary_conversion():
    import chex

    r = eigenvalue_to_binary(jnp.array([-1, 1]))

    chex.assert_trees_all_close(r, jnp.array([1, 0]))

    r = binary_to_eigenvalue(jnp.array([0, 1]))

    chex.assert_trees_all_close(r, jnp.array([1, -1]))


def test_expectation_value_to_eigenvalue():
    import chex

    result = expectation_value_to_eigenvalue(
        jnp.array([[1, -1, 0], [-1, 0, 1]]), SHOTS=10
    )

    expect_result = jnp.array(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]
    )

    assert jnp.array_equal(result, expect_result)

    # Hard core test
    expvals = jnp.linspace(-1, 1, 1001)
    result = expectation_value_to_eigenvalue(expvals, SHOTS=1000)

    chex.assert_trees_all_close(expvals, result.mean(axis=-1), atol=1e-08)


@pytest.mark.parametrize(
    "models",
    [
        (
            sq.models.nnx.WoModel([8], [8], rngs=nnx.Rngs(0)),
            sq.models.observable_to_expvals,
            random_nnx_module,
        ),
        (
            sq.models.nnx.UnitaryModel([8, 8], rngs=nnx.Rngs(0)),
            sq.models.toggling_unitary_to_expvals,
            random_nnx_module,
        ),
        (
            sq.models.linen.UnitaryModel([10, 10]),
            sq.models.toggling_unitary_to_expvals,
            random_flax_module,
        ),
        (
            sq.models.linen.WoModel([5], [5]),
            sq.models.observable_to_expvals,
            random_flax_module,
        ),
    ],
)
def test_save_and_load_model(tmp_path, load_dataset, models):
    loaded_data, train_data, test_data = load_dataset

    base_model, adapter_fn, flax_module = models

    graybox_model = sq.probabilistic.make_flax_probabilistic_graybox_model(
        name="graybox",
        base_model=base_model,
        adapter_fn=adapter_fn,
        prior=sq.probabilistic.dist.Normal(0, 1),
        flax_module=flax_module,
    )

    model = sq.probabilistic.make_probabilistic_model(
        predictive_model=graybox_model,
    )

    guide = sq.probabilistic.auto_diagonal_normal_guide(
        model,
        train_data.control_params,
        train_data.unitaries,
        train_data.observables,
        key=jax.random.key(0),
    )

    NUM_STEPS = 10
    optimizer = sq.optimize.get_default_optimizer(NUM_STEPS)

    svi = SVI(
        model=model,
        guide=guide,
        optim=numpyro.optim.optax_to_numpyro(optimizer),
        loss=TraceMeanField_ELBO(),
    )

    svi_state = svi.init(
        rng_key=jax.random.key(0),
        control_parameters=train_data.control_params,
        unitaries=train_data.unitaries,
        observables=train_data.observables,
    )

    update_fn = sq.probabilistic.make_update_fn(
        svi,
        control_parameters=train_data.control_params,
        unitaries=train_data.unitaries,
        observables=train_data.observables,
    )

    eval_fn = sq.probabilistic.make_evaluate_fn(
        svi,
        control_parameters=test_data.control_params,
        unitaries=test_data.unitaries,
        observables=test_data.observables,
    )

    eval_losses = []
    losses = []
    for i in range(NUM_STEPS):
        svi_state, loss = jax.jit(update_fn)(svi_state)
        eval_loss = jax.jit(eval_fn)(svi_state)
        losses.append(loss)
        eval_losses.append(eval_loss)

    svi_result = sq.probabilistic.SVIRunResult(
        svi.get_params(svi_state), svi_state, jnp.stack(losses), jnp.stack(eval_losses)
    )

    shots = loaded_data.experiment_data.experiment_config.shots

    result = sq.model.ModelData(
        params=svi_result.params,
        config={
            "shots": shots,
            "model": "WoBased",
            "model_config": {"hidden_sizes": "hidden_sizes"},
        },
    )

    path = tmp_path / "test"

    result.to_file(path / "model.json")

    read_result = sq.model.ModelData.from_file(path / "model.json")

    chex.assert_trees_all_close(read_result.params, result.params)
    assert result == read_result


def test_batched_matmul():
    # 1. Typical model training, control_params.shape == (batch, feature)
    # Expected weights and bias of shape (feature_in, out) and (out)
    # Expected expectation value of shape (batch, out)
    x = jnp.zeros((100, 4))
    w = jnp.zeros((4, 5))
    b = jnp.zeros((5))

    y = batched_matmul(x, w, b)
    assert y.shape == (100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    chex.assert_trees_all_close(x @ w + b, batched_matmul(x, w, b))

    # 2. During BOED

    # 2.1 Model should handle control_params.shape == (extra, design[batch], feature)
    # Model is vecterized by numpyro.plate_stack("plate", control_params.shape[:-1]):
    # Expected weights and bias of shape (extra, design[batch], feature_in, out) and (extra, design[batch], out)
    # Expected expectation value of shape (extra, design[batch], out)
    x = jnp.zeros((10, 100, 4))
    w = jnp.zeros((10, 100, 4, 5))
    b = jnp.zeros((10, 100, 5))

    y = batched_matmul(x, w, b)
    assert y.shape == (10, 100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    def dense(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (4,) and w.shape == (4, 5) and b.shape == (5,)
        return x @ w + b

    expected_y = jax.vmap(jax.vmap(dense, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(
        x, w, b
    )

    chex.assert_trees_all_close(expected_y, batched_matmul(x, w, b))

    # 2.2 Marginal guide handle control_params.shape == (extra, design[batch], feature)
    # Model is vecterized by the design[batch] only
    # Expected weights and bias of shape (design[batch], feature_in, out) and (design[batch], out)
    # Expected expectation value of shape (extra, design[batch], out)
    x = jnp.zeros((10, 100, 4))
    w = jnp.zeros((100, 4, 5))
    b = jnp.zeros((100, 5))

    y = batched_matmul(x, w, b)
    assert y.shape == (10, 100, 5)

    key = jax.random.key(0)
    x_key, w_key, b_key = jax.random.split(key, 3)
    x = jax.random.uniform(x_key, x.shape)
    w = jax.random.uniform(w_key, w.shape)
    b = jax.random.uniform(b_key, b.shape)

    expected_y = jax.vmap(jax.vmap(dense, in_axes=(0, 0, 0)), in_axes=(0, None, None))(
        x, w, b
    )

    chex.assert_trees_all_close(expected_y, batched_matmul(x, w, b))


def test_dense_layer():
    # Test normal inference
    for batch_size in [
        (10,),
        (
            20,
            10,
        ),
    ]:
        in_feature = 4
        out_feature = 5
        name = "test_dense"
        trace = get_trace(dense_layer)(
            jnp.ones(batch_size + (in_feature,)), name, in_feature, out_feature
        )

        expected_sites = [
            {
                "name": f"{name}.kernel",
                "type": "sample",
                "value_shape": (4, 5),
                "event_shape": (4, 5),
            },
            {
                "name": f"{name}.bias",
                "type": "sample",
                "value_shape": (5,),
                "event_shape": (5,),
            },
        ]

        for expected_site, site in zip(expected_sites, trace.values(), strict=True):
            assert expected_site["name"] == site["name"]
            assert expected_site["type"] == site["type"]
            assert expected_site["value_shape"] == site["value"].shape
            assert expected_site["event_shape"] == site["fn"].event_shape


def make_WoBased_bnn_model(
    name: str,
    shared_layers: tuple[int, ...] = (),
    pauli_layers: tuple[int, ...] = (),
    pauli_operators: tuple[str, ...] = ("X", "Y", "Z"),
    NUM_UNITARY_PARAMS: int = 3,
    NUM_DIAGONAL_PARAMS: int = 2,
    priors_fn: typing.Callable[
        [str, tuple[int, ...]], dist.Distribution
    ] = sq.probabilistic.default_priors_fn,
    unitary_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: 2
    * jnp.pi
    * jax.nn.hard_sigmoid(x),
    diagonal_activation_fn: typing.Callable[[jnp.ndarray], jnp.ndarray] = lambda x: (
        2 * jax.nn.hard_sigmoid(x)
    )
    - 1,
) -> typing.Callable:
    """Function to create Blackbox BNN with custom activation functions for unitary and diagonal output

    Args:
        unitary_activation_fn: Activation function for unitary parameters.
        diagonal_activation_fn: Activation function for diagonal parameters.
        priors_fn: Function to generate priors for parameters.

    Returns:
        Callable: Blackbox BNN model function
    """

    def model(
        x: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        # Main trunk network
        shared_x = x
        for i, hidden_size in enumerate(shared_layers):
            shared_x = dense_layer(
                shared_x,
                f"{name}/shared.dense_{i}",
                shared_x.shape[-1],
                hidden_size,
                priors_fn,
            )
            shared_x = jax.nn.relu(shared_x)

        Wos: dict[str, jnp.ndarray] = dict()

        for op in pauli_operators:
            # Branch network for each Pauli operator
            branch_x = jnp.copy(shared_x)

            # Sub hidden layers for this operator
            for i, hidden_size in enumerate(pauli_layers):
                branch_x = dense_layer(
                    branch_x,
                    f"{name}/pauli_{op}.dense_{i}",
                    branch_x.shape[-1],
                    hidden_size,
                    priors_fn,
                )
                branch_x = jax.nn.relu(branch_x)

            # Unitary parameters output
            unitary_params = dense_layer(
                branch_x,
                f"{name}/U_{op}",
                branch_x.shape[-1],
                NUM_UNITARY_PARAMS,
                priors_fn,
            )
            unitary_params = unitary_activation_fn(unitary_params)

            # Diagonal parameters output
            diag_params = dense_layer(
                branch_x,
                f"{name}/D_{op}",
                branch_x.shape[-1],
                NUM_DIAGONAL_PARAMS,
                priors_fn,
            )
            diag_params = diagonal_activation_fn(diag_params)

            # Combine into Wo using your existing function
            Wos[op] = sq.model.hermitian(unitary_params, diag_params)

        numpyro.deterministic(f"{name}/Wo", Wos)  # type: ignore
        return Wos

    return model


def test_probabilistic_graybox():
    test_input_1 = jnp.ones((10, 4))
    test_input_2 = jnp.ones((2, 10, 4))

    for test_input in [test_input_1, test_input_2]:
        batch_shape = test_input.shape[:-1]

        blackbox_bnn = make_WoBased_bnn_model("graybox")
        out = sq.probabilistic.get_trace(blackbox_bnn)(test_input)["graybox/Wo"][
            "value"
        ]

        expvals = sq.models.observable_to_expvals(
            out, jnp.broadcast_to(jnp.eye(2), batch_shape + (2, 2))
        )

        assert expvals.shape == batch_shape + (18,)


def get_data_model(
    detune: float, trotterization: bool = False, trotter_steps: int = 1000
) -> sq.utils.SyntheticDataModel:
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_gaussian_control_sequence(
        qubit_info=qubit_info
    )
    dt = 2 / 9

    ideal_hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.make_signal_fn(
            get_envelope=sq.control.get_envelope_transformer(
                control_sequence=control_sequence
            ),
            drive_frequency=qubit_info.frequency,
            dt=dt,
        ),
    )

    total_hamiltonian = sq.predefined.detune_x_hamiltonian(
        ideal_hamiltonian, detune * qubit_info.frequency
    )

    def ode_solver(hamiltonian):
        return sq.predefined.get_single_qubit_whitebox(
            hamiltonian=hamiltonian,
            control_sequence=control_sequence,
            qubit_info=qubit_info,
            dt=dt,
        )

    def trotter_solver(hamiltonian):
        return sq.physics.make_trotterization_solver(
            hamiltonian=hamiltonian,
            total_dt=control_sequence.total_dt,
            trotter_steps=trotter_steps,
            dt=dt,
            y0=jnp.eye(2, dtype=jnp.complex128),
        )

    solver = ode_solver if not trotterization else trotter_solver

    return sq.utils.SyntheticDataModel(
        control_sequence=control_sequence,
        qubit_information=qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=total_hamiltonian,
        solver=solver(total_hamiltonian),
        quantum_device=None,
        whitebox=solver(ideal_hamiltonian),
    )


def probabilistic_graybox_model(
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    priors_fn=sq.probabilistic.default_priors_fn,
):
    # Inner broadcast.
    samples_shape = control_parameters.shape[:-2]
    unitaries = jnp.broadcast_to(unitaries, samples_shape + unitaries.shape[-3:])

    model = make_WoBased_bnn_model("graybox", priors_fn=priors_fn)

    output = model(control_parameters)

    expectation_values = sq.models.observable_to_expvals(output, unitaries)

    return expectation_values


def marginal_guide(
    control_parameters: jnp.ndarray,  # Shape: (expanded_batch, candidate_design, features)
    unitaries: jnp.ndarray,
    observation_labels: list[str],
    target_labels: list[str],
    SEPARATE_OBSERVABLE: bool = False,
):
    """
    Marginal guide for the probabilistic graybox model.

    Args:
        control_parameters: Shape (expanded_batch, candidate_design, features)
        unitaries: Unitary matrices for expectation value computation
        observation_labels: Labels for observed variables
        target_labels: Labels for target variables
    """
    # Get the number of candidate designs
    num_candidates = control_parameters.shape[-2]

    # Get the graybox model structure - you'll need to specify these based on your actual model
    shared_layers = ()  # Adjust based on your model
    pauli_layers = ()  # Adjust based on your model
    pauli_operators = ("X", "Y", "Z")
    NUM_UNITARY_PARAMS = 3
    NUM_DIAGONAL_PARAMS = 2

    # Create variational parameters for each candidate design
    with numpyro.plate("candidates", num_candidates):
        # Create variational parameters for all the dense layers in the graybox model
        variational_params = {}

        # Shared/trunk network parameters
        current_features = control_parameters.shape[-1]
        for i, hidden_size in enumerate(shared_layers):
            layer_name = f"graybox/shared/Dense_{i}"

            # Kernel parameters
            kernel_loc = numpyro.param(
                f"{layer_name}.kernel_loc",
                jnp.zeros((num_candidates, current_features, hidden_size)),
            )
            kernel_scale = numpyro.param(
                f"{layer_name}.kernel_scale",
                jnp.ones((num_candidates, current_features, hidden_size)),
                constraint=dist.constraints.positive,
            )

            # Bias parameters
            bias_loc = numpyro.param(
                f"{layer_name}.bias_loc", jnp.zeros((num_candidates, hidden_size))
            )
            bias_scale = numpyro.param(
                f"{layer_name}.bias_scale",
                jnp.ones((num_candidates, hidden_size)),
                constraint=dist.constraints.positive,
            )

            # Sample parameters
            variational_params[f"{layer_name}.kernel"] = numpyro.sample(
                f"{layer_name}.kernel",
                dist.Normal(kernel_loc, kernel_scale).to_event(2),  # type: ignore
            )
            variational_params[f"{layer_name}.bias"] = numpyro.sample(
                f"{layer_name}.bias",
                dist.Normal(bias_loc, bias_scale).to_event(1),  # type: ignore
            )

            current_features = hidden_size

        # Branch network parameters for each Pauli operator
        shared_features = (
            current_features if shared_layers else control_parameters.shape[-1]
        )

        for op in pauli_operators:
            current_features = shared_features

            # Branch hidden layers
            for i, hidden_size in enumerate(pauli_layers):
                layer_name = f"graybox/Pauli_{op}/Dense_{i}"

                kernel_loc = numpyro.param(
                    f"{layer_name}.kernel_loc",
                    jnp.zeros((num_candidates, current_features, hidden_size)),
                )
                kernel_scale = numpyro.param(
                    f"{layer_name}.kernel_scale",
                    jnp.ones((num_candidates, current_features, hidden_size)),
                    constraint=dist.constraints.positive,
                )
                bias_loc = numpyro.param(
                    f"{layer_name}.bias_loc", jnp.zeros((num_candidates, hidden_size))
                )
                bias_scale = numpyro.param(
                    f"{layer_name}.bias_scale",
                    jnp.ones((num_candidates, hidden_size)),
                    constraint=dist.constraints.positive,
                )

                variational_params[f"{layer_name}.kernel"] = numpyro.sample(
                    f"{layer_name}.kernel",
                    dist.Normal(kernel_loc, kernel_scale).to_event(2),  # type: ignore
                )
                variational_params[f"{layer_name}.bias"] = numpyro.sample(
                    f"{layer_name}.bias",
                    dist.Normal(bias_loc, bias_scale).to_event(1),  # type: ignore
                )

                current_features = hidden_size

            # Unitary output layer
            unitary_layer_name = f"graybox/U_{op}"
            kernel_loc = numpyro.param(
                f"{unitary_layer_name}.kernel_loc",
                jnp.zeros((num_candidates, current_features, NUM_UNITARY_PARAMS)),
            )
            kernel_scale = numpyro.param(
                f"{unitary_layer_name}.kernel_scale",
                jnp.ones((num_candidates, current_features, NUM_UNITARY_PARAMS)),
                constraint=dist.constraints.positive,
            )
            bias_loc = numpyro.param(
                f"{unitary_layer_name}.bias_loc",
                jnp.zeros((num_candidates, NUM_UNITARY_PARAMS)),
            )
            bias_scale = numpyro.param(
                f"{unitary_layer_name}.bias_scale",
                jnp.ones((num_candidates, NUM_UNITARY_PARAMS)),
                constraint=dist.constraints.positive,
            )

            variational_params[f"{unitary_layer_name}.kernel"] = numpyro.sample(
                f"{unitary_layer_name}.kernel",
                dist.Normal(kernel_loc, kernel_scale).to_event(2),  # type: ignore
            )
            variational_params[f"{unitary_layer_name}.bias"] = numpyro.sample(
                f"{unitary_layer_name}.bias",
                dist.Normal(bias_loc, bias_scale).to_event(1),  # type: ignore
            )

            # Diagonal output layer
            diag_layer_name = f"graybox/D_{op}"
            kernel_loc = numpyro.param(
                f"{diag_layer_name}.kernel_loc",
                jnp.zeros((num_candidates, current_features, NUM_DIAGONAL_PARAMS)),
            )
            kernel_scale = numpyro.param(
                f"{diag_layer_name}.kernel_scale",
                jnp.ones((num_candidates, current_features, NUM_DIAGONAL_PARAMS)),
                constraint=dist.constraints.positive,
            )
            bias_loc = numpyro.param(
                f"{diag_layer_name}.bias_loc",
                jnp.zeros((num_candidates, NUM_DIAGONAL_PARAMS)),
            )
            bias_scale = numpyro.param(
                f"{diag_layer_name}.bias_scale",
                jnp.ones((num_candidates, NUM_DIAGONAL_PARAMS)),
                constraint=dist.constraints.positive,
            )

            variational_params[f"{diag_layer_name}.kernel"] = numpyro.sample(
                f"{diag_layer_name}.kernel",
                dist.Normal(kernel_loc, kernel_scale).to_event(2),  # type: ignore
            )
            variational_params[f"{diag_layer_name}.bias"] = numpyro.sample(
                f"{diag_layer_name}.bias",
                dist.Normal(bias_loc, bias_scale).to_event(1),  # type: ignore
            )

        # Forward pass through the graybox model
        expvals = forward_pass_graybox(
            control_parameters,
            unitaries,
            variational_params,
            shared_layers,
            pauli_layers,
            pauli_operators,
        )

        # Sample observations
        if SEPARATE_OBSERVABLE:
            for idx, exp in enumerate(sq.constant.default_expectation_values_order):
                numpyro.sample(
                    f"obs/{exp.initial_state}/{exp.observable}",
                    dist.BernoulliProbs(
                        probs=sq.utils.expectation_value_to_prob_minus(
                            jnp.expand_dims(expvals[..., idx], axis=-1)
                        )
                    ).to_event(1),  # type: ignore
                )
        else:
            probs = sq.utils.expectation_value_to_prob_minus(expvals)
            numpyro.sample(
                "obs",
                dist.BernoulliProbs(probs=probs).to_event(1),  # type: ignore
                infer={"enumerate": "parallel"},
            )


def forward_pass_graybox(
    control_parameters: jnp.ndarray,
    unitaries: jnp.ndarray,
    params: dict,
    shared_layers: tuple[int, ...],
    pauli_layers: tuple[int, ...],
    pauli_operators: tuple[str, ...],
    # Unitary activation functions
    unitary_activation_fn=lambda x: 2 * jnp.pi * jax.nn.hard_sigmoid(x),
    diagonal_activation_fn=lambda x: (2 * jax.nn.hard_sigmoid(x)) - 1,
) -> jnp.ndarray:
    """Forward pass through the graybox model using sampled parameters."""

    # Inner broadcast for unitaries
    samples_shape = control_parameters.shape[:-2]
    unitaries = jnp.broadcast_to(unitaries, samples_shape + unitaries.shape[-3:])

    # Shared trunk network
    shared_x = control_parameters
    for i, hidden_size in enumerate(shared_layers):
        layer_name = f"graybox/shared/Dense_{i}"
        W = params[f"{layer_name}.kernel"]
        b = params[f"{layer_name}.bias"]
        shared_x = sq.probabilistic.batched_matmul(shared_x, W, b)
        shared_x = jax.nn.relu(shared_x)

    Wos = {}
    for op in pauli_operators:
        # Branch network
        branch_x = jnp.copy(shared_x)

        for i, hidden_size in enumerate(pauli_layers):
            layer_name = f"graybox/Pauli_{op}/Dense_{i}"
            W = params[f"{layer_name}.kernel"]
            b = params[f"{layer_name}.bias"]
            branch_x = sq.probabilistic.batched_matmul(branch_x, W, b)
            branch_x = jax.nn.relu(branch_x)

        # Unitary parameters
        unitary_layer_name = f"graybox/U_{op}"
        W_u = params[f"{unitary_layer_name}.kernel"]
        b_u = params[f"{unitary_layer_name}.bias"]
        unitary_params = sq.probabilistic.batched_matmul(branch_x, W_u, b_u)
        unitary_params = unitary_activation_fn(unitary_params)

        # Diagonal parameters
        diag_layer_name = f"graybox/D_{op}"
        W_d = params[f"{diag_layer_name}.kernel"]
        b_d = params[f"{diag_layer_name}.bias"]
        diag_params = sq.probabilistic.batched_matmul(branch_x, W_d, b_d)
        diag_params = diagonal_activation_fn(diag_params)

        # Combine into Wo
        Wos[op] = sq.models.hermitian(unitary_params, diag_params)

    # Convert to expectation values
    return sq.models.observable_to_expvals(Wos, unitaries)


def test_boed():
    FAST_MODE = True
    n_opitimization_steps = 1000 if not FAST_MODE else 10
    data_model = get_data_model(detune=0.001, trotterization=True)
    assert callable(data_model.whitebox)
    control_parameters_candidate_designs = jnp.linspace(
        0.0, 2 * jnp.pi, num=500
    ).reshape(-1, 1)

    unitaries_candidate_designs = jax.vmap(data_model.whitebox)(
        control_parameters_candidate_designs
    )[:, -1, :, :]

    eig_key = jax.random.key(0)
    SEPARATE_OBSERVABLE = True
    if SEPARATE_OBSERVABLE:
        observation_labels = [
            f"obs/{exp.initial_state}/{exp.observable}"
            for exp in sq.constant.default_expectation_values_order
        ]
    else:
        observation_labels = ["obs"]

    eig, state = sq.boed.estimate_eig(
        eig_key,
        sq.boed.vectorized_for_eig(
            (
                sq.probabilistic.make_probabilistic_model(
                    probabilistic_graybox_model,
                    separate_observables=SEPARATE_OBSERVABLE,
                )
            )
        ),
        partial(marginal_guide, SEPARATE_OBSERVABLE=SEPARATE_OBSERVABLE),
        sq.predefined.polynomial_feature_map(control_parameters_candidate_designs, 4),
        unitaries_candidate_designs,
        optimizer=sq.optimize.get_default_optimizer(n_iterations=n_opitimization_steps),
        num_optimization_steps=n_opitimization_steps,
        observation_labels=observation_labels,
        target_labels=[],
        num_particles=1000 if not FAST_MODE else 10,
        final_num_particles=1_000 if not FAST_MODE else 1_000,
        callbacks=[],
    )
