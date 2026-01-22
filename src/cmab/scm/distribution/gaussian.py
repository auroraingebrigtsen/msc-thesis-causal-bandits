from .base import BaseDistribution

class Gaussian(BaseDistribution):
    def __init__(self, mu: float, sigma: float):
        super().__init__()
        self.mu0 = float(mu)
        self.sigma0 = float(sigma)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, rng) -> float:
        return rng.normal(self.mu, self.sigma)
    
    def expected(self):
        return self.mu
    
    def distribution_shift(self, rng, max_delta: float) -> None:
        delta = rng.uniform(-max_delta, max_delta)
        new_mu = self.mu + delta
        self.mu = new_mu

    def reset(self) -> None:
        self.mu = self.mu0
        self.sigma = self.sigma0