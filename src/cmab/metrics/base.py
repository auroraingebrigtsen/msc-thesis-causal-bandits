from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, reward: float):
        pass

    @abstractmethod
    def get_regrets(self) -> list:
        pass

    @abstractmethod
    def get_regret_at_step(self, step: int) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass