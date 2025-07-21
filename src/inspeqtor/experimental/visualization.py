import matplotlib.pyplot as plt
import jax.numpy as jnp
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from .constant import default_expectation_values_order


def format_expectation_values(
    expvals: jnp.ndarray,
) -> dict[str, dict[str, jnp.ndarray]]:
    """This function formats expectation values of shape (18, N) to a dictionary
    with the initial state as outer key and the observable as inner key.

    Args:
        expvals (jnp.ndarray): Expectation values of shape (18, N). Assumes that order is as in default_expectation_values_order.

    Returns:
        dict[str, dict[str, jnp.ndarray]]: A dictionary with the initial state as outer key and the observable as inner key.
    """
    expvals_dict: dict[str, dict[str, jnp.ndarray]] = {}
    for idx, exp in enumerate(default_expectation_values_order):
        if exp.initial_state not in expvals_dict:
            expvals_dict[exp.initial_state] = {}

        expvals_dict[exp.initial_state][exp.observable] = expvals[idx]

    return expvals_dict


def plot_expectation_values(
    expvals_dict: dict[str, dict[str, jnp.ndarray]],
    title: str,
):
    fig, axes = plt.subplot_mosaic(
        """
        +r0
        -l1
        """,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
    )

    colormap = {
        "X": "#ef4444",
        "Y": "#6366f1",
        "Z": "#10b981",
    }

    for idx, (initial_state, expvals) in enumerate(expvals_dict.items()):
        ax = axes[initial_state]
        for observable, expval in expvals.items():
            ax.plot(expval, "-", label=observable, color=colormap[observable])
        ax.set_title(f"Initial state: {initial_state}")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(loc="upper left")

    # Set title for the figure
    fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


def plot_loss_with_moving_average(
    x: jnp.ndarray | np.ndarray,
    y: jnp.ndarray | np.ndarray,
    ax: Axes,
    window: int = 50,
    annotate_at: list[int] = [2000, 4000, 6000, 8000, 10000],
    **kwargs,
) -> Axes:
    """Plot the moving average of the given argument y

    Args:
        x (jnp.ndarray | np.ndarray): The horizontal axis
        y (jnp.ndarray | np.ndarray): The vertical axis
        ax (Axes): Axes object
        window (int, optional): The moving average window. Defaults to 50.
        annotate_at (list[int], optional): The list of x positions to annotate the y value. Defaults to [2000, 4000, 6000, 8000, 10000].

    Returns:
        Axes: Axes object.
    """
    moving_average = pd.Series(np.asarray(y)).rolling(window=window).mean()

    ax.plot(
        x,
        moving_average,
        **kwargs,
    )

    # Annotate the loss values every 1000 iterations
    for idx in annotate_at:
        i = idx - 1
        loss_value = y[i]
        ax.annotate(
            f"{loss_value:.2f}",
            xy=(x[i].item(), y[i].item()),
            xytext=(-10, 10),  # 10 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    return ax


def assert_list_of_axes(axes) -> list[Axes]:
    """Assert the provide object that they are a list of Axes

    Args:
        axes (typing.Any): Expected to be numpy array of Axes

    Returns:
        list[Axes]: The list of Axes
    """
    assert isinstance(axes, np.ndarray)
    axes = axes.flatten()

    for ax in axes:
        assert isinstance(ax, Axes)
    return axes.tolist()


def set_fontsize(ax: Axes, fontsize: float | int):
    """Set all fontsize of the Axes object

    Args:
        ax (Axes): The Axes object which fontsize to be changed.
        fontsize (float | int): The fontsize.
    """
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)

    legend, handles = ax.get_legend_handles_labels()

    ax.legend(legend, handles, fontsize=fontsize)
