from .base import BasePmf
import numpy as np

class BernoulliPmf(BasePmf):
    def __init__(self, p: float,):
        super().__init__()
        self.p0 = float(p)
        self.p = float(p)

    def sample(self, rng) -> int:
        return int(rng.binomial(1, self.p))

    def mean(self) -> float:
        return self.p

    def distribution_shift(self, rng, max_delta: float) -> None:
        delta = rng.uniform(-max_delta, max_delta)
        new_p = self.p + delta
        self.p = min(1.0, max(0.0, new_p))

    def reset(self) -> None:
        self.p = self.p0