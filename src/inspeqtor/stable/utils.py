from inspeqtor.v2.utils import (
    SyntheticDataModel as SyntheticDataModel,
    single_qubit_shot_quantum_device as single_qubit_shot_quantum_device,
    calculate_expectation_values as calculate_expectation_values,
    dictorization as dictorization,
    count_bits as count_bits,
    finite_shot_quantum_device as finite_shot_quantum_device,
    get_measurement_probability as get_measurement_probability,
    finite_shot_expectation_value as finite_shot_expectation_value,
    tensor_product as tensor_product,
)

from inspeqtor.v2.data import check_parity as check_parity

from inspeqtor.v1.utils import (
    random_split as random_split,
    dataloader as dataloader,
    variance_of_observable as variance_of_observable,
    expectation_value_to_prob_plus as expectation_value_to_prob_plus,
    expectation_value_to_prob_minus as expectation_value_to_prob_minus,
    expectation_value_to_eigenvalue as expectation_value_to_eigenvalue,
    eigenvalue_to_binary as eigenvalue_to_binary,
    binary_to_eigenvalue as binary_to_eigenvalue,
    recursive_vmap as recursive_vmap,
    calculate_shots_expectation_value as calculate_shots_expectation_value,
    enable_jax_x64 as enable_jax_x64,
    disable_jax_x64 as disable_jax_x64,
)

from inspeqtor.v2.constant import (
    default_expectation_values_order as default_expectation_values_order,
    get_default_expectation_values_order as get_default_expectation_values_order,
    SX as SX,
    X as X,
    Y as Y,
    Z as Z,
    plus_projectors as plus_projectors,
    minus_projectors as minus_projectors,
)

from inspeqtor.v1.visualization import (
    plot_control_envelope as plot_control_envelope,
    plot_expectation_values as plot_expectation_values,
    assert_list_of_axes as assert_list_of_axes,
    set_fontsize as set_fontsize,
    plot_loss_with_moving_average as plot_loss_with_moving_average,
)

from dataclasses import dataclass


@dataclass
class Colors:
    red: str = "#f43f5e"
    blue: str = "#6366f1"
    orange: str = "#f97316"
    green: str = "#22c55e"
    gray: str = "#a2a2a2"
    yellow: str = "#ffe019"


colors = Colors()
