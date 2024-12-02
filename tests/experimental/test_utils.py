import jax
import jax.numpy as jnp
from inspeqtor.experimental.utils import dataloader


def test_dataloader(DATA_SIZE: int = 100, BATCH_SIZE: int = 15, NUM_EPOCHS: int = 10):
    train_key = jax.random.key(0)

    x_mock = jnp.linspace(0, 10, DATA_SIZE).reshape(-1, 1)
    y_mock = jnp.sin(x_mock)

    # Expected number of batches per epoch
    num_batches = x_mock.shape[0] // BATCH_SIZE
    if x_mock.shape[0] % BATCH_SIZE != 0:
        num_batches += 1

    expected_final_batch_idx = num_batches - 1
    expected_step = num_batches * NUM_EPOCHS - 1

    step = 0
    batch_idx = 0
    is_last_batch = True
    epoch_idx = 0

    for (step, batch_idx, is_last_batch, epoch_idx), (x_batch, y_batch) in dataloader(
        (x_mock, y_mock), batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, key=train_key
    ):
        print(
            f"step: {step}, batch_idx: {batch_idx}, is_last_batch: {is_last_batch}, epoch_idx: {epoch_idx}, x_batch: {x_batch.shape}, y_batch: {y_batch.shape}"
        )

    assert step == expected_step
    assert batch_idx == expected_final_batch_idx
    assert is_last_batch
    assert epoch_idx == NUM_EPOCHS - 1
