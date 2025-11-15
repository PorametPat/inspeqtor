import jax.numpy as jnp
import typing
from flax import linen as nn
from flax.typing import VariableDict

from inspeqtor.experimental.models.linen import (
    WoModel as WoModel,
    UnitaryModel as UnitaryModel,
    UnitarySPAMModel as UnitarySPAMModel,
    train_model as train_model,
    make_predictive_fn as make_predictive_fn,
    create_step as create_step,
)


def make_loss_fn(
    adapter_fn: typing.Callable,
    model: nn.Module,
    evaluate_fn: typing.Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
):
    """_summary_

    Args:
        predictive_fn (typing.Callable): Function for calculating expectation value from the model
        model (nn.Module): Flax linen Blackbox part of the graybox model.
        evaluate_fn ( typing.Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray, , jnp.ndarray]): Take in predicted and experimental expectation values and ideal unitary and return loss value
    """

    def loss_fn(
        params: VariableDict,
        control_parameters: jnp.ndarray,
        unitaries: jnp.ndarray,
        expectation_values: jnp.ndarray,
        **model_kwargs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """This function implement a unified interface for nn.Module.

        Args:
            params (VariableDict): Model parameters to be optimized
            control_parameters (jnp.ndarray): Control parameters parametrized Hamiltonian
            unitaries (jnp.ndarray): The Ideal unitary operators corresponding to the control parameters
            expectation_values (jnp.ndarray): Experimental expectation values to calculate the loss value

        Returns:
            tuple[jnp.ndarray, dict[str, jnp.ndarray]]: The loss value and other metrics.
        """
        output = model.apply(params, control_parameters, **model_kwargs)
        predicted_expectation_value = adapter_fn(output, unitaries=unitaries)

        loss = evaluate_fn(predicted_expectation_value, expectation_values, unitaries)

        return (loss, {})

    return loss_fn
