from .base import BaseDistribution
from cmab.scm.domain.interval import IntervalDomain

class Uniform(BaseDistribution):
    def __init__(self, domain: IntervalDomain):
        super().__init__()
        self.domain = domain

    def sample(self, rng) -> int:
        return int(rng.uniform(self.domain.lower, self.domain.upper + 1))
    
    def expected_value(self) -> float:
        return (self.domain.lower + self.domain.upper) / 2
    
    def prob(self, x: int) -> float:
        if self.domain.lower <= x <= self.domain.upper:
            return 1 / (self.domain.upper - self.domain.lower + 1) 
        else:
            return 0.0
        
    def update_parameters(self, new_params: dict[str, float]) -> None:
        pass

    def support(self) -> list[int]:
        return list(range(self.domain.lower, self.domain.upper + 1))

    def reset(self) -> None:
        pass