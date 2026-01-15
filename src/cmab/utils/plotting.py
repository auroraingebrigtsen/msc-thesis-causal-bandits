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