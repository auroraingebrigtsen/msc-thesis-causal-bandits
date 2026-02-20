import matplotlib.pyplot as plt

def plot_regrets(regrets, labels, title):
    for regret, label in zip(regrets, labels):
        plt.plot(regret, label=label)
    plt.xlabel("Time Steps")
    plt.ylabel("Averaged Cumulative Regret")
    plt.title(title)
    plt.legend()
    plt.savefig("regret_curve.png", dpi=300, bbox_inches="tight")
    plt.close()



def plot_regrets_and_change_points(regrets, labels, title, change_points: list, T:int):
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
    plt.savefig("regret_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


from typing import Dict, List, Sequence, Optional
import numpy as np

def plot_detection_rate_heatmap(
    change_points: List[Dict[str, List[int]]],
    true_change_points: Optional[Sequence[int]] = None,
    title: str = "Detection-rate heatmap over time (by node)",
) -> tuple[np.ndarray, list[str]]:

    n_runs = len(change_points)

    # Collect node set
    all_nodes = set()
    for run_dict in change_points:
        all_nodes.update(run_dict.keys())

    nodes = sorted(all_nodes)

    # Infer T (and validate)
    # Find first available series to infer length
    T = None
    for run_dict in change_points:
        for node in nodes:
            if node in run_dict:
                T = len(run_dict[node])
                break
        if T is not None:
            break

    # Validate and aggregate
    counts = np.zeros((len(nodes), T), dtype=np.float64)
    

    plt.tight_layout()
    plt.savefig("detection_rate_heatmap")
    plt.close(fig)