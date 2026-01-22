from .base import BaseCPD

class SimpleCPD(BaseCPD):
    """Naive CPD that flags a change if the newest reward deviates from past mean by > threshold."""
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.rewards: list[float] = []
        self.num_pulls = 0

    def _is_change_point(self) -> bool:
        if len(self.rewards) < 2:
            return False

        prev = self.rewards[:-1]
        prev_mean = sum(prev) / len(prev)
        return abs(self.rewards[-1] - prev_mean) > self.threshold

    def update(self, reward: float) -> None:
        self.rewards.append(reward)
        self.num_pulls += 1

    def reset(self) -> None:
        self.rewards.clear()
        self.num_pulls = 0
