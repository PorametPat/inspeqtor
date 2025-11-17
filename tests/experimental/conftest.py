import jax
import jax.numpy as jnp
import inspeqtor.experimental as sq
import logging
import pytest

logging.basicConfig(level=logging.INFO)

TROTTER_STEPS = 1_000


@pytest.fixture(scope="session")
def generate_dataset():
    logging.info("Generate synthetic dataset")

    data_model = sq.predefined.get_predefined_data_model_m1()
    key = jax.random.key(0)
    # Now, we use the noise model to performing the data using simulator.
    exp_data, _, _, _ = sq.predefined.generate_experimental_data(
        key=key,
        hamiltonian=data_model.total_hamiltonian,
        sample_size=100,
        strategy=sq.predefined.SimulationStrategy.SHOT,
        get_qubit_information_fn=lambda: data_model.qubit_information,
        get_control_sequence_fn=lambda: data_model.control_sequence,
        method=sq.predefined.WhiteboxStrategy.TROTTER,
        trotter_steps=TROTTER_STEPS,
    )

    return data_model, exp_data


@pytest.fixture(scope="session")
def load_dataset(generate_dataset):
    logging.info("Load dataset")
    random_split_key = jax.random.key(0)

    data_model, exp_data = generate_dataset

    # Now we can prepare the dataset that ready to use.
    whitebox = sq.physics.make_trotterization_solver(
        data_model.ideal_hamiltonian,
        data_model.control_sequence.total_dt,
        data_model.dt,
        trotter_steps=TROTTER_STEPS,
        y0=jnp.eye(2, dtype=jnp.complex128),
    )
    loaded_data = sq.utils.prepare_data(exp_data, data_model.control_sequence, whitebox)

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
        loaded_data.observed_values,
    )
    train_data = sq.data.DataBundled(
        control_params=sq.predefined.drag_feature_map(train_control_parameters),
        unitaries=train_unitaries,
        observables=train_expectation_values,
    )

    test_data = sq.data.DataBundled(
        control_params=sq.predefined.drag_feature_map(test_control_paramaeters),
        unitaries=test_unitaries,
        observables=test_expectation_values,
    )
    return loaded_data, train_data, test_data
