from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update_parameters(self, param_updates: dict[str, float]) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

