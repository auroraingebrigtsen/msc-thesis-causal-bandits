# Base class for all bandit algorithms

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class BaseBanditAlgorithm(ABC):
    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def _update(self, arm, reward):
        pass
    
    @abstractmethod
    def reset(self):
        pass