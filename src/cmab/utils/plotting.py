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