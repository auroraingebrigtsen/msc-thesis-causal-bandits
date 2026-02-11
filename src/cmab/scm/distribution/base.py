from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class BasePMF(BaseDistribution):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def prob(self, x):
        pass

    @abstractmethod
    def support(self):
        pass

    @abstractmethod
    def reset(self, seed: int):
        pass


class BasePDF(BaseDistribution):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def reset(self, seed: int):
        pass