# Rliable bare bones code for neuroevobench evaluation
from typing import List
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

colors = sns.color_palette("colorblind")


IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)


def plot_rliable(
    score_dict: dict,
    agg_fn: str = "all",
    reps: int = 50000,
):
    """Create a rliable plot from a score dictionary."""
    if agg_fn == "median":
        aggregate_func = lambda x: np.array([MEDIAN(x)])
        metric_names = ["Median"]
    elif agg_fn == "iqm":
        aggregate_func = lambda x: np.array([IQM(x)])
        metric_names = ["Interquartile Mean (IQM)"]
    elif agg_fn == "mean":
        aggregate_func = lambda x: np.array([MEAN(x)])
        metric_names = ["Mean"]
    elif agg_fn == "og":
        aggregate_func = lambda x: np.array([OG(x)])
        metric_names = ["Optimality Gap"]
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=reps
    )
    algorithms = list(score_dict.keys())
    fig, axes = plot_interval_estimates(
        aggregate_scores,
        aggregate_interval_estimates,
        metric_names=metric_names,
        algorithms=algorithms,
        xlabel_y_coordinate=-0.16,
        xlabel="OpenES Norm. Performance",
    )
    fig.tight_layout()
    return fig, axes


def plot_interval_estimates(
    point_estimates,
    interval_estimates,
    metric_names,
    algorithms=None,
    colors=None,
    color_palette="colorblind",
    max_ticks=4,
    xlabel="Normalized Score",
    **kwargs
):
    """Plots various metrics with confidence intervals.

    Args:
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metrics to plot.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      metric_names: Names of the metrics corresponding to `point_estimates`.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
        `plt.MaxNLocator`.
      subfigure_width: Width of each subfigure.
      row_height: Height of each row in a subfigure.
      xlabel_y_coordinate: y-coordinate of the x-axis label.
      xlabel: Label for the x-axis.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      fig: A matplotlib Figure.
      axes: `axes.Axes` or array of Axes.
    """

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])
    # figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    figsize = (12, 20)
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(
            color_palette, n_colors=len(algorithms)
        )
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop("interval_height", 0.6)

    for alg_idx, algorithm in enumerate(algorithms):
        ax = axes
        # Plot interval estimates.
        lower, upper = interval_estimates[algorithm][:, 0]
        ax.barh(
            y=alg_idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=colors[algorithm],
            alpha=0.75,
            label=algorithm,
        )
        # Plot point estimates.
        ax.vlines(
            x=point_estimates[algorithm][0],
            ymin=alg_idx - (7.5 * h / 16),
            ymax=alg_idx + (6 * h / 16),
            label=algorithm,
            color="k",
            alpha=0.5,
        )

    ax.set_yticks(list(range(len(algorithms))))
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax.set_yticklabels(algorithms, fontsize="x-large")
    ax.set_title(metric_names[0], fontsize="xx-large")
    ax.tick_params(axis="both", which="major")
    _decorate_axis(ax, ticklabelsize="x-large", wrect=5)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlabel(xlabel, fontsize="x-large")
    # fig.text(0.4, xlabel_y_coordinate, xlabel, ha="center", fontsize="xx-large")
    plt.subplots_adjust(wspace=kwargs.pop("wspace", 0.11), left=0.0)
    return fig, axes


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax
