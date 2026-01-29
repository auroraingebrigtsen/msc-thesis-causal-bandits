from itertools import product
from cmab.typing import InterventionSet, Observation
from .base import BaseCPD


class PageHinkleyCPD(BaseCPD):
    """Very simple CPD estimator + naive change detection for binary SCMs."""

    def __init__(self, node: str, parents: list[str], delta: float = 0.005, lambda_: float = 50, min_samples_for_detection: int = 10):
        # PH test parameters
        self.delta = delta  # tolerance for PH test
        self.lambda_ = lambda_  # threshold for PH test
        self.min_samples_for_detection = min_samples_for_detection

        self.node = node
        self.parents = parents

        # parent config-> Dict of PH state: {'t': int, 'mean': float, 'm_t': float, 'M_t': float}
        self.ph_state = self._initialize_ph_state()


    def _initialize_ph_state(self) -> dict[tuple[int, ...], dict[str, float]]:
        ph_state = {}
        parent_cfgs = list(product([0, 1], repeat=len(self.parents)))  # Currently only binary variables, TODO: extend to use domains
        for cfg in parent_cfgs:
            ph_state[cfg] = {'t': 0, 'mean': 0.0, 'm_t': 0.0, 'M_t': 0.0}
        return ph_state


    def update(self, observation: Observation) -> bool:
            cfg = tuple(int(observation[p]) for p in self.parents)

            x = observation[self.node]

            # Online PH update
            state = self.ph_state[cfg]

            state['t'] += 1
            prev_mean = state['mean']
            mean = prev_mean + (x - prev_mean) / state['t'] # running mean

            state['mean'] = mean
            state['m_t'] += (x - mean - self.delta) # cumulative deviation
            state['M_t'] = min(state['M_t'], state['m_t'])  # minimum deviation

            # compute PH statistic
            ph_t = state['m_t'] - state['M_t']

            if state['t'] >= self.min_samples_for_detection and ph_t > self.lambda_:
                return True
        
            return False
    

    def reset(self) -> None:
        """Resets the PH state."""
        self.ph_state = self._initialize_ph_state()
        