# Base class for all bandit algorithms

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class BaseBanditAlgorithm(ABC):
    def __init__(self, n_arms):
        self.n_arms = n_arms

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def _update(self, chosen_arm, reward):
        pass
    
    @abstractmethod
    def reset(self):
        pass