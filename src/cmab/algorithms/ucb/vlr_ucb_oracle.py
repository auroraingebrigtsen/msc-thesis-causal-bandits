from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
import numpy as np

class OracleVlrUCBAgent(PomisUCBAgent):
    def __init__(
        self,
        reward_node: str,
        G: CausalDiagram,
        arms: list[Intervention],
        c: float,  # UCB exploration parameter
        atomic: bool,
        changed_vars: list = [],
        change_points = [],
    ):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.means = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)

        self.changed_vars = changed_vars
        self.change_points = change_points
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis

    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        if self.t in self.change_points:
            arm_index = self.means.argmax() # In oracle mode, we reset the one with highest mean, as this is the one most likely to have changed (as it was the best performing one)
            reset_arm = self.arms[arm_index]
            intervention_set = frozenset(var for var, _ in reset_arm)
            for a in self.arms:
                if frozenset(var for var, _ in a) == intervention_set:
                    a_index = self.arm_to_index[a]
                    self.reset_arm(a_index)

    def reset_arm(self, arm_index: int) -> None:
        self.means[arm_index] = 0.0
        self.arm_samples[arm_index] =0.0
        self.cpds[arm_index] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
        self.resat_arms[self.arms[arm_index]].append(self.t)

            
    def reset(self) -> None:
        super().reset()
        self.resat_arms = {arm : [] for arm in self.arms}