# Base class for all bandit algorithms

from abc import ABC, abstractmethod
from cmab.typing import InterventionSet, Observation

class BaseBanditAlgorithm(ABC):
    def __init__(self, reward_node: str):
        self.reward_node = reward_node

    @abstractmethod
    def select_arm(self) -> InterventionSet:
        pass

    @abstractmethod
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass