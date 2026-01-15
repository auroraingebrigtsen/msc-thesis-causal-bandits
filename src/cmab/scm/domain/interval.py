from .base import FiniteDiscreteDomain

class IntervalDomain(FiniteDiscreteDomain):
    def __init__(self, lower: int, upper: int):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def support(self) -> list:
        return list(range(self.lower, self.upper + 1))

    def contains(self, value: int) -> bool:
        return self.lower <= value <= self.upper