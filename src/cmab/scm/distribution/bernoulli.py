from .base import BaseDistribution

class Bernoulli(BaseDistribution):
    def __init__(self, p: float,):
        super().__init__()
        self.p0 = p
        self.p = p

    def sample(self, rng) -> int:
        return int(rng.binomial(1, self.p))
    
    def expected_value(self) -> float:
        return self.p

    def prob(self, x: int) -> float:
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        else:
            return 0.0
        
    def update_parameters(self, new_params: dict[str, float]) -> None:
        """Update the parameters of the distribution based on the provided updates."""
        if "p" in new_params:
            self.p = new_params["p"]

    def support(self) -> list[int]:
        return [0, 1]

    def reset(self) -> None:
        self.p = self.p0