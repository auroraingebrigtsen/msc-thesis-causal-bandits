from .base import FiniteDiscreteDomain

class BinaryDomain(FiniteDiscreteDomain):
    def __init__(self):
        super().__init__()
        self.values = [0, 1]

    def support(self) -> list:
        return self.values

    def contains(self, value: int) -> bool:
        return value in self.values