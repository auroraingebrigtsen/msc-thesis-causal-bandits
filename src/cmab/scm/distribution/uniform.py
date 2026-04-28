from .base import BaseDistribution

class Uniform(BaseDistribution):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.n = len(values)

    def sample(self, rng):
        return rng.choice(self.values)
    
    def expected_value(self) -> float:
        return sum(self.values) / self.n
    
    def prob(self, x: float) -> float:
        if x in self.values:
            return 1 / self.n
        else:
            return 0.0
        
    def update_parameters(self, new_params: dict[str, float]) -> None:
        pass

    def support(self) -> list[int]:
        return self.values

    def reset(self) -> None:
        pass