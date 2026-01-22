from abc import ABC, abstractmethod


class BaseCPD(ABC):
    def __init__(self,):
        pass    
    
    @abstractmethod
    def  _is_change_point(self) -> bool:
        pass

    @abstractmethod
    def update(self, reward: float) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

