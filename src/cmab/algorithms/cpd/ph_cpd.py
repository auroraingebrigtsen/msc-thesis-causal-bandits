from itertools import product
from cmab.typing import InterventionSet, Observation
from .base import BaseCPD
from river import drift


class PageHinkleyCPD(BaseCPD):
    """Very simple CPD estimator + naive change detection for binary SCMs. 
    Made for node-level CPD."""

    def __init__(self, node: str, parents: list[str], delta: float = 0.05, lambda_: float = 5, min_samples_for_detection: int = 10):
        # PH test parameters
        self.delta = delta  # tolerance for PH test
        self.lambda_ = lambda_  # threshold for PH test
        self.min_samples_for_detection = min_samples_for_detection

        self.node = node
        self.parents = parents

        self.ph_state = self._initialize_ph_state()


    def _initialize_ph_state(self) -> dict[tuple[int, ...], dict[str, float]]:
        ph_state = {}
        ph = {}
        parent_cfgs = list(product([0, 1], repeat=len(self.parents)))  # Currently only binary variables, TODO: extend to use domains
        for cfg in parent_cfgs:
            ph_state[cfg] = {'t': 0, 'mean': 0.0, 'm_pos': 0.0, 'M_pos': 0.0, 'm_neg': 0.0, 'M_neg': 0.0}
            ph[cfg] = drift.PageHinkley()
        return ph_state
    
    def update(self, observation: Observation) -> bool:
         cfg = tuple(int(observation[p]) for p in self.parents)
         self.ph[cfg].update(observation[self.node])
         return self.ph[cfg].drift_detected

    def _update(self, observation: Observation) -> bool:
            cfg = tuple(int(observation[p]) for p in self.parents)
            print(f"Updating CPD for node {self.node} with parent configuration {cfg} and observation {observation[self.node]}")  # Debug print

            x = observation[self.node]

            # Online PH update
            state = self.ph_state[cfg]

            state['t'] += 1
            prev_mean = state['mean']
            mean = prev_mean + (x - prev_mean) / state['t'] # running mean
            state['mean'] = mean

            state['m_pos'] += x - mean + self.delta # cumulative positive deviation
            state['M_pos'] = min(state['M_pos'], state['m_pos'])  # minimum positive deviation
            state['m_neg'] += mean - x + self.delta # cumulative negative deviation
            state['M_neg'] = max(state['M_neg'], state['m_neg'])  # minimum negative deviation

            # compute PH statistic
            ph_pos = state['m_pos'] - state['M_pos']
            ph_neg = state['m_neg'] - state['M_neg']

            print(f"PH state for node {self.node}, cfg {cfg}: t={state['t']}, mean={state['mean']}, m_pos={state['m_pos']}, M_pos={state['M_pos']}, m_neg={state['m_neg']}, M_neg={state['M_neg']}, ph_pos={ph_pos}, ph_neg={ph_neg}")  # Debug print

            if state['t'] >= self.min_samples_for_detection and (ph_pos > self.lambda_ or ph_neg > self.lambda_):
                return True
        
            return False

    def reset(self) -> None:
        """Resets the PH state."""
        self.ph_state = self._initialize_ph_state()
        


class ArmLevelPageHinkleyCPD(BaseCPD):
    """Very simple CPD estimator + naive change detection for binary SCMs. 
    Made for arm-level CPD with two sided PH test"""

    def __init__(self, delta: float, lambda_:float, min_samples_for_detection: int):
        # PH test parameters
        self.delta = delta  # tolerance for PH test
        self.lambda_ = lambda_  # threshold for PH test
        self.min_samples_for_detection = min_samples_for_detection
        self.ph_state = {'t': 0, 'mean': 0.0, 'm_pos': 0.0, 'M_pos': 0.0, 'm_neg': 0.0, 'M_neg': 0.0}


    def update(self, reward: float) -> bool:
            # Online PH update
            self.ph_state['t'] += 1
            prev_mean = self.ph_state['mean']
            mean = prev_mean + (reward - prev_mean) / self.ph_state['t'] # running mean
            self.ph_state['mean'] = mean

            self.ph_state['m_pos'] += (reward - mean ) - self.delta # cumulative positive deviation
            self.ph_state['M_pos'] = min(self.ph_state['M_pos'], self.ph_state['m_pos'])  # minimum positive deviation
            self.ph_state['m_neg'] += (mean - reward) - self.delta # cumulative negative deviation
            self.ph_state['M_neg'] = min(self.ph_state['M_neg'], self.ph_state['m_neg'])  # minimum negative deviation

            # compute PH statistic
            ph_pos = self.ph_state['m_pos'] - self.ph_state['M_pos']
            ph_neg = self.ph_state['m_neg'] - self.ph_state['M_neg']

            if self.ph_state['t'] >= self.min_samples_for_detection and (ph_pos > self.lambda_ or ph_neg > self.lambda_):
                return True
        
            return False
    

    def reset(self) -> None:
        """Resets the PH state."""
        self.ph_state = {'t': 0, 'mean': 0.0, 'm_pos': 0.0, 'M_pos': 0.0, 'm_neg': 0.0, 'M_neg': 0.0}
        