# Base class for finite discrete domains, which is the values that SCM variables can take as intervention values. 
# TODO: not sure if they should be called domains in this sense

from abc import ABC

class FiniteDiscreteDomain(ABC):
    def __init__(self):
        pass

    def support(self) -> list:
        pass

    def contains(self, value: float) -> bool:
        pass