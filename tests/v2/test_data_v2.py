import jax
import polars as pl
import numpy as np
from flax.traverse_util import flatten_dict

import inspeqtor.experimental as sq
from inspeqtor.v2.data import ExperimentConfiguration, ExperimentalData
from inspeqtor.v2.predefined import (
    get_predefined_data_model_m1,
    save_data_to_path,
    load_data_from_path,
)
from inspeqtor.v2.control import ravel_unravel_fn


def test_ExperimentConfig(tmp_path):
    qubit_info = sq.predefined.get_mock_qubit_information()

    experiment_config = ExperimentConfiguration(
        qubits=[qubit_info],
        expectation_values_order=sq.constant.default_expectation_values_order,
        parameter_structure=[("0", "param1"), ("1", "param2")],
        backend_name="qasm_simulator",
        sample_size=2,
        shots=1000,
        EXPERIMENT_IDENTIFIER="0001",
        EXPERIMENT_TAGS=["test", "test2"],
        description="This is a test experiment",
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=10,
        instance="open",
    )

    # To dict
    dict_experiment_config = experiment_config.to_dict()

    # From dict to dataclass
    experiment_config_from_dict = ExperimentConfiguration.from_dict(
        dict_experiment_config
    )

    assert experiment_config == experiment_config_from_dict

    d = tmp_path / "test"
    d.mkdir()

    # Test to_file()
    experiment_config.to_file(path=d)

    # Test from_file()
    experiment_config_from_file = ExperimentConfiguration.from_file(path=d)

    assert experiment_config == experiment_config_from_file


def test_ExperimentalData(tmp_path):
    data_model = get_predefined_data_model_m1()
    seq = data_model.control_sequence

    config = ExperimentConfiguration(
        qubits=[data_model.qubit_information],
        expectation_values_order=sq.constant.default_expectation_values_order,
        parameter_structure=seq.get_structure(),
        backend_name="inspeqtor",
        shots=1000,
        EXPERIMENT_IDENTIFIER="0001",
        EXPERIMENT_TAGS=[],
        device_cycle_time_ns=2 / 9,
        sequence_duration_dt=320,
        description="From the test of ExperimentalData",
        sample_size=10,
        instance="inspeqtor",
    )

    key = jax.random.key(0)
    sample_key, device_key = jax.random.split(key)

    ravel_fn, _ = ravel_unravel_fn(seq)
    # Sample the parameter by vectorization.
    params_dict = jax.vmap(seq.sample_params)(
        jax.random.split(sample_key, config.sample_size)
    )
    # Prepare parameter in single line
    params = jax.vmap(ravel_fn)(params_dict)

    # Perform experiment locally
    expvals = sq.utils.shot_quantum_device(
        device_key, params, solver=data_model.solver, SHOTS=config.shots
    )

    param_df = pl.DataFrame(
        jax.tree.map(lambda x: np.array(x), flatten_dict(params_dict, sep="/"))
    ).with_row_index("parameter_id")

    obs_df = pl.DataFrame(
        jax.tree.map(
            lambda x: np.array(x),
            flatten_dict(
                sq.visualization.format_expectation_values(expvals.T), sep="/"
            ),
        )
    ).with_row_index("parameter_id")

    exp_data = ExperimentalData(config, param_df, obs_df)

    path = tmp_path / "test"
    path.mkdir()

    exp_data.save_to_folder(path)
    reread_exp_data = exp_data.from_folder(path)

    assert exp_data == reread_exp_data

    path = tmp_path / "data"
    path.mkdir()

    save_data_to_path(path, experiment_data=exp_data, control_sequence=seq)

    loaded_data = load_data_from_path(
        path,
        hamiltonian_spec=sq.predefined.HamiltonianSpec(
            method=sq.predefined.WhiteboxStrategy.TROTTER,
            hamiltonian_enum=sq.predefined.HamiltonianEnum.rotating_transmon_hamiltonian,
            trotter_steps=10_000,
        ),
    )

    assert loaded_data.experiment_data == exp_data
    assert loaded_data.control_parameters.shape == (config.sample_size, 2)
    assert loaded_data.unitaries.shape == (config.sample_size, 2, 2)
    assert loaded_data.observed_values.shape == (config.sample_size, 18)
