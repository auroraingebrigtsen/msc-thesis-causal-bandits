# Base class for all bandit algorithms

from abc import ABC, abstractmethod

class BaseBanditAlgorithm(ABC):
    def __init__(self, n_arms):
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update(self, chosen_arm, reward):
        pass

    @abstractmethod
    def reset(self):
        pass