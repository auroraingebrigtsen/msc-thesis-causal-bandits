from .base import BaseDistribution

class Bernoulli(BaseDistribution):
    def __init__(self, p: float,):
        super().__init__()
        self.p0 = p
        self.p = p

    def sample(self, rng) -> int:
        return int(rng.binomial(1, self.p))

    def expected(self):
        return self.p
    
    def prob(self, x: int) -> float:
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        else:
            return 0.0
        
    def update_parameters(self, param_updates: dict[str, float]) -> None:
        """Update the parameters of the distribution based on the provided updates."""
        if "p" in param_updates:
            self.p = param_updates["p"]

    def reset(self) -> None:
        self.p = self.p0