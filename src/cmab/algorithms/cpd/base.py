from abc import ABC, abstractmethod


class BaseCPD(ABC):
    def __init__(self,):
        pass    

    @abstractmethod
    def update(self, reward: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

