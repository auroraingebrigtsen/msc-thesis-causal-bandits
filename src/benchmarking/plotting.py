import matplotlib.pyplot as plt
from cmab.typing import Intervention
import numpy as np
import seaborn as sns

def _format_arm(arm: Intervention) -> str:
    return "do(" + ", ".join(f"{var}={val}" for var, val in arm) + ")"

def plot_regrets_and_change_points(regrets, labels, change_points: list, T: int,
                                   std_devs=None,
                                   save_path="plots/regret_curve_with_cps.png"):
    """
    Plots averaged cumulative regrets with optional confidence bands and change point markers.

    Args:
        regrets (list of np.ndarray): Averaged cumulative regret arrays (shape: [T] each).
        labels (list of str): Labels for each regret array.
        title (str): Plot title.
        change_points (list of int): Time steps where change points occur.
        T (int): Total number of time steps.
        std_devs (list of np.ndarray, optional): Std dev arrays for each agent, for shaded bands.
        save_path (str): Output path for the saved figure.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    palette = sns.color_palette("deep", n_colors=len(labels))
    colors = [palette[i] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (regret, label) in enumerate(zip(regrets, labels)):
        x = np.arange(len(regret))
        ax.plot(x, regret, label=label, color=colors[i], linewidth=2)
        if std_devs is not None:
            ax.fill_between(x,
                            regret - std_devs[i],
                            regret + std_devs[i],
                            color=colors[i], alpha=0.15)

    for idx, t in enumerate(change_points):
        ax.axvline(x=t, color="slategray", linestyle="--", linewidth=1.2, alpha=0.7, label="Change point" if idx == 0 else None)
        ax.text(t+1, ax.get_ylim()[1]*0.95, f"CP{idx+1}", color="slategray", fontsize=9, va="top")

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Averaged Cumulative Regret")
    ax.set_title(f"Averaged Cumulative Regret over Horizon {T}")
    ax.legend(framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_reset_rate_heatmap(
    reset_counts: dict[Intervention, np.ndarray],
    title: str = "Reset-rate heatmap over time (by arm)",
    save_path: str = "plots/reset_rate_heatmap.png"
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
    ax.set_yticklabels([_format_arm(a) for a in arms])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Reset count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)

def plot_means(
    means_history: dict,
    save_path: str = "plots/historical_means.png",
    change_points: list[int] | None = None,
):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    fig, ax = plt.subplots(figsize=(9, 5))

    palette = sns.color_palette("Set1", n_colors=len(means_history))
    for (arm, means), color in zip(means_history.items(), palette):
        ax.plot(means, label=f"Arm: {_format_arm(arm)}", linewidth=2,
                color=color, marker="o", markevery=500, markersize=6)

    for idx, cp in enumerate(change_points or []):
        ax.axvline(x=cp, color="slategray", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="Change point" if idx == 0 else None)
        ax.text(cp+1, 1.0, f"CP{idx+1}", color="slategray", fontsize=9, va="top", ha="center",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Reward-means per arm over horizon T={len(next(iter(means_history.values())))}")
    ax.legend(framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)