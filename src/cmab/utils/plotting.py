import matplotlib.pyplot as plt
from cmab.typing import InterventionSet
import numpy as np

def plot_regrets(regrets, labels, title):
    for regret, label in zip(regrets, labels):
        plt.plot(regret, label=label)
    plt.xlabel("Time Steps")
    plt.ylabel("Averaged Cumulative Regret")
    plt.title(title)
    plt.legend()
    plt.savefig("regret_curve.png", dpi=300, bbox_inches="tight")
    plt.close()



def plot_regrets_and_change_points(regrets, labels, title, change_points: list, T:int, save_path="regret_curve_with_cps.png"):
    """Plots the averaged cumulative regrets along with vertical lines indicating change points.
    Args:
        regrets (list of np.ndarray): List of averaged cumulative regret arrays for each agent.
        labels (list of str): List of labels corresponding to each regret array.
        title (str): Title of the plot.
        change_points (list of int): List of time steps where change points occur.
    """
    for regret, label in zip(regrets, labels):
        plt.plot(regret, label=label)
    for t in range(1, T):
        if t in change_points:
            plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)

    plt.xlabel("Time Steps")
    plt.ylabel("Averaged Cumulative Regret")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def plot_reset_rate_heatmap(
    reset_counts: dict[InterventionSet, np.ndarray],
    title: str = "Reset-rate heatmap over time (by arm)",
    save_path: str = "reset_rate_heatmap.png"
):

    arms = list(reset_counts.keys())
    series = [np.asarray(reset_counts[a]).ravel() for a in arms]

    data = np.vstack(series)  # (n_arms, T)

    fig, ax = plt.subplots(figsize=(12, 4 + 0.25 * len(arms)))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Arm")

    # Label arms on y-axis (stringified)
    ax.set_yticks(np.arange(len(arms)))
    ax.set_yticklabels([str(a) for a in arms])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reset count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)