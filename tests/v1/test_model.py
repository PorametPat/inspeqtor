import jax
import inspeqtor.v1 as sq
from flax import nnx


def test_check_allclose():
    tree = {"a": jax.numpy.array([1.0, 2.0]), "b": jax.numpy.array([3.0, 4.0])}
    tree_copy = tree.copy()

    # Use tree.map to apply jax.numpy.allclose to each leaf of the tree
    res = jax.tree.map(lambda x, y: jax.numpy.allclose(x, y), tree, tree_copy)

    if not jax.tree_util.tree_all(res):
        raise AssertionError("The trees are not all close.")


def test_linen_model(load_dataset, tmp_path):
    # Initialization

    loaded_data, train_data, test_data = load_dataset

    model = sq.models.linen.WoModel(
        shared_layers=[10],
        pauli_layers=[10],
    )

    # Working with adapter
    loss_fn = sq.models.linen.make_loss_fn(
        adapter_fn=sq.models.observable_to_expvals,
        model=model,
        evaluate_fn=lambda x, y, z: sq.models.mse(x, y),
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

    train_loss, _ = loss_fn(
        model_params,  # type: ignore
        train_data.control_params,
        train_data.unitaries,
        train_data.observables,
    )
    assert train_loss.shape == ()

    # Working prediction_fn
    predictive_fn = sq.models.linen.make_predictive_fn(
        sq.models.observable_to_expvals,
        model,
        model_params,  # type: ignore
    )

    _, l2a_fn = sq.control.get_param_array_converter(loaded_data.control_sequence)
    sample_params = l2a_fn(loaded_data.control_sequence.sample_params(params_key))

    unitary_f = loaded_data.whitebox(sample_params)[-1]
    predictive_fn(sq.predefined.drag_feature_map(sample_params), unitary_f)

    model_data = sq.model.ModelData(
        model_params,
        {
            "shared_layers": [10],
            "pauli_layers": [10],
        },
    )

    model_data.to_file(tmp_path / "test_model_data.json")
    model_data_from_file = sq.model.ModelData.from_file(
        tmp_path / "test_model_data.json"
    )

    assert model_data == model_data_from_file


def test_nnx_model(load_dataset, tmp_path):
    # Initialization

    loaded_data, train_data, test_data = load_dataset

    model = sq.models.nnx.WoModel(shared_layers=[8], pauli_layers=[8], rngs=nnx.Rngs(0))

    # Working with adapter
    loss_fn = sq.models.nnx.make_loss_fn(
        adapter_fn=sq.models.observable_to_expvals,
        evaluate_fn=lambda x, y, z: sq.models.mse(x, y),
    )

    train_loss, _ = loss_fn(model, train_data)
    assert train_loss.shape == ()

    key = jax.random.key(0)
    training_key, params_key = jax.random.split(key)
    NUM_EPOCH = 10
    optimizer = sq.optimize.get_default_optimizer(8 * NUM_EPOCH)
    _, opt_state, histories = sq.models.nnx.train_model(
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
        sq.models.observable_to_expvals,
        model,
    )

    _, l2a_fn = sq.control.get_param_array_converter(loaded_data.control_sequence)
    sample_params = l2a_fn(loaded_data.control_sequence.sample_params(params_key))

    unitary_f = loaded_data.whitebox(sample_params)[-1]
    predictive_fn(sq.predefined.drag_feature_map(sample_params), unitary_f)

    _, state = nnx.split(model)
    model_params = nnx.to_pure_dict(state)

    model_data = sq.model.ModelData(
        model_params,
        {
            "shared_layers": [8],
            "pauli_layers": [8],
        },
    )

    model_data.to_file(tmp_path / "test_nnx_model_data.json")
    model_data_from_file = sq.model.ModelData.from_file(
        tmp_path / "test_nnx_model_data.json"
    )

    assert model_data == model_data_from_file
