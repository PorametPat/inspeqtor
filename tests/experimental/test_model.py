import jax
import inspeqtor.experimental as sq
import pytest
from flax import nnx
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session")
def generate_dataset():
    # caplog.set_level(logging.INFO)

    logging.info("Generate synthetic dataset")

    trotter_steps = 1_000

    data_model = sq.predefined.get_predefined_data_model_m1()

    # Now, we use the noise model to performing the data using simulator.
    exp_data, _, _, _ = sq.predefined.generate_experimental_data(
        key=jax.random.key(0),
        hamiltonian=data_model.total_hamiltonian,
        sample_size=100,
        strategy=sq.predefined.SimulationStrategy.SHOT,
        get_qubit_information_fn=lambda: data_model.qubit_information,
        get_control_sequence_fn=lambda: data_model.control_sequence,
        method=sq.predefined.WhiteboxStrategy.TROTTER,
        trotter_steps=trotter_steps,
    )

    # Now we can prepare the dataset that ready to use.
    whitebox = sq.physics.make_trotterization_solver(
        data_model.ideal_hamiltonian,
        data_model.control_sequence,
        data_model.dt,
        trotter_steps=trotter_steps,
    )
    loaded_data = sq.utils.prepare_data(exp_data, data_model.control_sequence, whitebox)
    key = jax.random.key(0)
    key, random_split_key, training_key = jax.random.split(key, 3)
    (
        train_control_parameters,
        train_unitaries,
        train_expectation_values,
        test_control_paramaeters,
        test_unitaries,
        test_expectation_values,
    ) = sq.utils.random_split(
        random_split_key,
        int(loaded_data.control_parameters.shape[0] * 0.1),  # Test size
        loaded_data.control_parameters,
        loaded_data.unitaries,
        loaded_data.expectation_values,
    )
    train_data = sq.optimize.DataBundled(
        control_params=sq.predefined.drag_feature_map(train_control_parameters),
        unitaries=train_unitaries,
        observables=train_expectation_values,
    )

    test_data = sq.optimize.DataBundled(
        control_params=sq.predefined.drag_feature_map(test_control_paramaeters),
        unitaries=test_unitaries,
        observables=test_expectation_values,
    )
    return loaded_data, train_data, test_data


def test_linen_model(generate_dataset):
    # Initialization

    loaded_data, train_data, test_data = generate_dataset

    model = sq.models.linen.WoModel(
        hidden_sizes_1=[10],
        hidden_sizes_2=[10],
    )

    # Working with adapter
    loss_fn = sq.models.linen.make_loss_fn(
        adapter_fn=sq.model.observable_to_expvals,
        model=model,
        calculate_metric_fn=sq.model.calculate_metric,
        loss_metric=sq.model.LossMetric.MSEE,
    )

    key = jax.random.key(0)
    training_key, params_key = jax.random.split(key)
    NUM_EPOCH = 10
    optimizer = sq.optimize.get_default_optimizer(8 * NUM_EPOCH)
    model_params, opt_state, histories = sq.models.linen.train_model(
        training_key,
        train_data=train_data,
        val_data=test_data,  # Here, we did not care about the validating dataset.
        test_data=test_data,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=[],
        NUM_EPOCH=NUM_EPOCH,
    )

    # Working prediction_fn
    predictive_fn = sq.models.linen.make_predictive_fn(
        sq.model.observable_to_expvals,
        model,
        model_params,  # type: ignore
    )

    _, l2a_fn = sq.control.get_param_array_converter(loaded_data.control_sequence)
    sample_params = l2a_fn(loaded_data.control_sequence.sample_params(params_key))

    unitary_f = loaded_data.whitebox(sample_params)[-1]
    predictive_fn(sq.predefined.drag_feature_map(sample_params), unitary_f)


def test_nnx_model(generate_dataset):
    # Initialization

    loaded_data, train_data, test_data = generate_dataset

    model = sq.models.nnx.WoModel(shared_layers=[8], pauli_layers=[8], rngs=nnx.Rngs(0))

    # Working with adapter
    loss_fn = sq.models.nnx.make_loss_fn(
        adapter_fn=sq.model.observable_to_expvals,
        calculate_metric_fn=sq.model.calculate_metric,
        loss_metric=sq.model.LossMetric.MSEE,
    )

    key = jax.random.key(0)
    training_key, params_key = jax.random.split(key)
    NUM_EPOCH = 10
    optimizer = sq.optimize.get_default_optimizer(8 * NUM_EPOCH)
    model_params, opt_state, histories = sq.models.nnx.train_model(
        training_key,
        train_data=train_data,
        val_data=test_data,  # Here, we did not care about the validating dataset.
        test_data=test_data,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=[],
        NUM_EPOCH=NUM_EPOCH,
    )

    # Working prediction_fn
    predictive_fn = sq.models.nnx.make_predictive_fn(
        sq.model.observable_to_expvals,
        model,
    )

    _, l2a_fn = sq.control.get_param_array_converter(loaded_data.control_sequence)
    sample_params = l2a_fn(loaded_data.control_sequence.sample_params(params_key))

    unitary_f = loaded_data.whitebox(sample_params)[-1]
    predictive_fn(sq.predefined.drag_feature_map(sample_params), unitary_f)
