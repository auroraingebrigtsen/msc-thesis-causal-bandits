import numpy as np
from .base import BaseMetric

class CumulativeRegret(BaseMetric):
    def __init__(self, T:int):
        self.T = T
        self.cumulative_regret = 0.0
        self.cumulative_regrets = np.zeros(T)
        self.step = 0

    def update(self, reward: float, optimal_expected_reward: float) -> None:
        """Updates the cumulative regret with the received reward."""
        instant_regret = optimal_expected_reward - reward
        self.cumulative_regret += instant_regret
        self.cumulative_regrets[self.step] = self.cumulative_regret
        self.step += 1
    
    def get_regrets(self) -> np.ndarray:
        """Returns the array of cumulative regrets recorded at each step."""
        return self.cumulative_regrets
    
    def get_regret_at_step(self, step:int) -> float:
        """Returns the cumulative regret at a specific step."""
        return self.cumulative_regrets[step]

    def reset(self) -> None:
        """Resets the cumulative regret metric."""
        self.cumulative_regret = 0.0
        self.cumulative_regrets = np.zeros(self.T)
        self.step = 0