import jax
import jax.numpy as jnp
import inspeqtor as sq


def test_generate_mock_data():
    data_model = sq.data.library.get_predefined_data_model_m1()

    sample_size = 10

    # Now, we use the noise model to performing the data using simulator.
    exp_data, _, _, _ = sq.data.library.generate_single_qubit_experimental_data(
        key=jax.random.key(0),
        hamiltonian=data_model.total_hamiltonian,
        sample_size=sample_size,
        strategy=sq.physics.library.SimulationStrategy.SHOT,
        qubit_inforamtion=data_model.qubit_information,
        control_sequence=data_model.control_sequence,
        method=sq.physics.library.WhiteboxStrategy.TROTTER,
        trotter_steps=1_000,
    )

    # Now we can prepare the dataset that ready to use.
    whitebox = sq.physics.make_trotterization_solver(
        data_model.ideal_hamiltonian,
        data_model.control_sequence.total_dt,
        data_model.dt,
        trotter_steps=1_000,
        y0=jnp.eye(2, dtype=jnp.complex_),
    )
    loaded_data = sq.data.prepare_data(exp_data, data_model.control_sequence, whitebox)

    assert loaded_data.control_parameters.shape == (sample_size, 2)
    assert loaded_data.unitaries.shape == (sample_size, 2, 2)
    assert loaded_data.observed_values.shape == (sample_size, 18)
