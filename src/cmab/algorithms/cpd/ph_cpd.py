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
        parent_cfgs = list(product([0, 1], repeat=len(self.parents)))  # Currently only binary variables, TODO: extend to use domains
        for cfg in parent_cfgs:
            ph_state[cfg] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
        return ph_state
    
    def update(self, observation: Observation) -> bool:
         cfg = tuple(int(observation[p]) for p in self.parents)
         print(f"Updating CPD for node {self.node} with parent configuration {cfg} and observation {observation[self.node]}")  # Debug print

         self.ph_state[cfg].update(observation[self.node])
         return self.ph_state[cfg].drift_detected


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
        self.ph_state = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)


    def update(self, reward: float) -> bool:
            # Online PH update
        self.ph_state.update(reward)
        return self.ph_state.drift_detected
    

    def reset(self) -> None:
        """Resets the PH state."""
        self.ph_state = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
        