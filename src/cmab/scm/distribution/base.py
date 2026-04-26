from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    def __init__(self):
        pass    
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def expected_value(self) -> float:
        """Return the expected value of the distribution."""
        pass

    @abstractmethod
    def prob(self, x: int) -> float:
        """Return the probability of a given value x."""
        pass

    @abstractmethod
    def update_parameters(self, new_params: dict[str, float]) -> None:
        pass

    @abstractmethod
    def support(self) -> list[int]:
        """Return the support of the distribution, i.e., the set of values with non-zero probability."""
        pass

    @abstractmethod
    def reset(self):
        pass

