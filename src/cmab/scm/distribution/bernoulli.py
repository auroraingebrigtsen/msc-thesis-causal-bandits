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

    def distribution_shift(self, rng, max_delta: float) -> None:
        delta = rng.uniform(-max_delta, max_delta)
        new_p = self.p + delta
        self.p = min(1.0, max(0.0, new_p))

    def reset(self) -> None:
        self.p = self.p0