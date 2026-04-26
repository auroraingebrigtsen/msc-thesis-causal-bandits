from .base import BaseDistribution

class Uniform(BaseDistribution):
    def __init__(self, lower: int, upper: int):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def sample(self, rng) -> int:
        return int(rng.uniform(self.lower, self.upper + 1))
    
    def expected_value(self) -> float:
        return (self.lower + self.upper) / 2
    
    def prob(self, x: int) -> float:
        if self.lower <= x <= self.upper:
            return 1 / (self.upper - self.lower + 1) 
        else:
            return 0.0
        
    def update_parameters(self, new_params: dict[str, float]) -> None:
        pass

    def support(self) -> list[int]:
        return list(range(self.lower, self.upper + 1))

    def reset(self) -> None:
        pass