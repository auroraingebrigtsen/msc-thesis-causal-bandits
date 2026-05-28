import matplotlib.pyplot as plt
from cmab.typing import Intervention
import numpy as np
import seaborn as sns

custom_palette = [
    "#0072B2",  
    "#E69F00",  
    "#009E73",  
    "#CC79A7", 
    "#D55E00", 
    "#56B4E9", 
    "#F0E442",  
    "#000000", 
    "#999999", 
    "#00429D",  
    "#73A2C6", 
    "#5A2928",  
    "#470142" ,
    "#025A02",  
    "#A59DEE",  
    "#AD6244"  
]

agent_palette = {
    "UCB": "#D55E00",
    "POMIS-UCB": "#D55E00",
    "PHT-UCB-sr": "#0072B2",
    "UCB-oracle-sr": "#56B4E9",
    "PHT-UCB-global": "#E69F00",
    "UCB-oracle-global": "#F0E442",
    "PHT-UCB-local": "#CC79A7",
    "PHT-VLR-UCB": "#33AA3D",
    "UCB-oracle-vlr": "#0B4410",
}

def _format_arm(arm: Intervention) -> str:
    return "do(" + ", ".join(f"{var}={val}" for var, val in arm) + ")"

def plot_regrets_and_change_points(regrets, labels, change_points, T,
                                   std_devs=None,
                                   save_path="plots/regret_curve_with_cps.png"):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    # Look up colors by label name instead of using sns palette
    colors = [agent_palette[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (regret, label) in enumerate(zip(regrets, labels)):
        x = np.arange(len(regret))
        ax.plot(x, regret, label=label, color=colors[i], linewidth=2)
        if std_devs is not None:
            ax.fill_between(
                x,
                regret - std_devs[label],
                regret + std_devs[label],
                color=colors[i],
                alpha=0.15
            )

    for idx, t in enumerate(change_points):
        ax.axvline(x=t, color="slategray", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="Change point" if idx == 0 else None)
        ax.text(t + 1, ax.get_ylim()[1] * 0.95, f"CP{idx+1}",
                color="slategray", fontsize=9, va="top")

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Averaged Cumulative Regret")
    ax.set_title(f"Averaged Cumulative Regret over Horizon T={T}")
    ax.legend(framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_detected_nodes_heatmap(
    detected_nodes: dict[str, np.ndarray],
    agent_name: str,
    save_path: str = "plots/detected_nodes_heatmap.png"
):
    nodes = list(detected_nodes.keys())
    series = [np.asarray(detected_nodes[node]).ravel() for node in nodes]

    data = np.vstack(series)  # (n_nodes, T)

    fig, ax = plt.subplots(figsize=(12, 4 + 0.25 * len(nodes)))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="turbo")

    ax.set_title(f"Detected Change Points by Node Over Time for Agent {agent_name}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Node")

    # Label nodes on y-axis
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_yticklabels(nodes)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_reset_rate_heatmap(
    reset_counts: dict[Intervention, np.ndarray],
    agent_name: str,
    save_path: str = "plots/reset_rate_heatmap.png"
):

    arms = list(reset_counts.keys())
    series = [np.asarray(reset_counts[a]).ravel() for a in arms]

    data = np.vstack(series)  # (n_arms, T)

    fig, ax = plt.subplots(figsize=(12, 4 + 0.25 * len(arms)))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="turbo")

    ax.set_title(f"Reset Count by Arm Over Time for Agent {agent_name}")
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
    plt.close(fig)

def plot_means(
    means_history: dict,
    save_path: str = "plots/historical_means.png",
    change_points: list[int] | None = None,
):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    fig, ax = plt.subplots(figsize=(9, 5))

    palette = sns.color_palette(custom_palette, n_colors=len(means_history))
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
    ax.set_title(f"Reward-means Per Arm Over Horizon T={len(next(iter(means_history.values())))}")
    ax.legend(framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)