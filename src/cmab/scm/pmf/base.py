from abc import ABC, abstractmethod
import numpy as np

class BasePmf(ABC):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def reset(self, seed: int):
        pass