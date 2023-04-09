from typing import Union
import matplotlib.pyplot as plt


def plot_task(
    results_dict,
    fig=None,
    ax=None,
    title: str = "Task Performance",
    xlabel: str = "Number of Generations",
    ylabel: str = "Performance",
    curve_labels: Union[list, None] = [
        "ARS",
        "OpenAI-ES",
        "PGPE",
        "SNES",
        "Sep-CMA-ES",
    ],
    legend_loc: int = 0,
    plot_ylabel: bool = True,
    plot_xlabel: bool = True,
    plot_legend: bool = True,
    colors: list = ["r", "g", "b", "yellow", "k"],
    linestyles: list = 10 * ["-"],
):
    """Performance plots - lcurve across generations."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # default_cycler = cycler(color=colors)

    plt.rc("lines", linewidth=3)
    # plt.rc("axes", prop_cycle=default_cycler)

    if curve_labels is None:
        i = 0
        for k, v in results_dict.items():
            ax.plot(v["time"], v["lcurve"], label=k, c=colors[i])
            i += 1
    else:
        for i, k in enumerate(results_dict):
            ax.plot(
                results_dict[k]["time"],
                results_dict[k]["lcurve"],
                label=curve_labels[i],
                c=colors[i],
                ls=linestyles[i],
            )

    # Prettify the plot
    if plot_legend:
        ax.legend(fontsize=14, loc=legend_loc)
    ax.set_title(title, fontsize=28)
    if plot_ylabel:
        ax.set_ylabel(ylabel, fontsize=24)
    if plot_xlabel:
        ax.set_xlabel(xlabel, fontsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=20)
    fig.tight_layout()
    return fig, ax
