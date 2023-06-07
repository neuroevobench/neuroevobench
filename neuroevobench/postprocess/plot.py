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
        ax.legend(loc=legend_loc, fontsize=14)
    ax.set_title(title)
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major")
    fig.tight_layout()
    return fig, ax


def plot_sensitivity(
    results_dict,
    fig=None,
    ax=None,
    title: str = "Hyperparameter Sensitivity",
    xlabel: str = "Evolutionary Optimizer",
    ylabel: str = "Performance",
    plot_ylabel: bool = True,
    plot_xlabel: bool = True,
    colors: list = ["r", "g", "b", "yellow", "k"],
    top_k: int = 50,
    search_budgets={"S": 10, "M": 40, "L": 50},
):
    """Hyperparameter sensitivity - best test scores."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    all_data = []
    all_labels = []
    for k, v in results_dict.items():
        for budget in ["S", "M", "L"]:
            sub_scores = v["max_scores"][: search_budgets[budget]]
            top_scores = sub_scores[sub_scores.argsort()][-top_k:][::-1]
            all_labels.append(k + "-" + budget)
            all_data.append(top_scores)

    bplot = ax.boxplot(
        all_data,
        notch=True,
        vert=True,
        patch_artist=True,
        labels=all_labels,
        widths=0.2,
    )

    i = 0
    for patch in bplot["boxes"]:
        # Split the label into strategy and budget
        strategy, budget = all_labels[i].split("-")
        patch.set_facecolor(colors[strategy])
        if budget == "S":
            patch.set_alpha(0.25)
        elif budget == "M":
            patch.set_alpha(0.5)
        elif budget == "L":
            patch.set_alpha(1.0)
        i += 1
    ax.set_title(title)

    ax.yaxis.grid(True)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=75, fontsize=15)
    else:
        ax.set_xticks([])
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    return


def plot_history(
    results_dict,
    fig=None,
    ax=None,
    title: str = "Hyperparameter Search - Best Test Scores",
    xlabel: str = "# Search Iterations",
    ylabel: str = "Performance",
    plot_ylabel: bool = True,
    plot_xlabel: bool = True,
    plot_legend: bool = True,
    colors: list = ["r", "g", "b", "yellow", "k"],
):
    """Hyperparameter sensitivity - best test scores."""
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    for k, v in results_dict.items():
        ax.plot(v, label=k, c=colors[k])

    # Add vertical lines to plot after 10, 40 trials
    ax.axvline(20, ls="--", alpha=0.75, c="grey")
    ax.axvline(40, ls="--", alpha=0.75, c="grey")

    ax.set_title(title)
    if plot_legend:
        ax.legend(loc=0, fontsize=16, ncol=2)
    if plot_xlabel:
        ax.set_xlabel(xlabel)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=30)
    else:
        ax.set_xticks([])
    if plot_ylabel:
        ax.set_ylabel(ylabel)
    return
