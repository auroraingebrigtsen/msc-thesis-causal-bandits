from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
import numpy as np

class OracleUCBAgent(PomisUCBAgent):
    def __init__(
        self,
        reward_node: str,
        G: CausalDiagram,
        arms: list[Intervention],
        c: float,  # UCB exploration parameter
        atomic: bool,
        reset_all: bool = True,  # whether to reset all arms or just the one that triggered the change point
        changed_vars: list = [],
        change_points = [],
    ):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.means = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.reset_all = reset_all

        self.changed_vars = changed_vars
        self.change_points = change_points
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis

    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        arm_index = self.arm_to_index[arm]

        if self.t in self.change_points:
            arm_index = self.means.argmax() # In oracle mode, we assume we know which arm changed, so we reset the one with highest mean, as this is the one most likely to have changed (as it was the best performing one)
            print(f"Step {self.t}: Change point detected for arm {arm}!")

            if self.reset_all: # Reset all arms
                self.means = np.zeros(self.n_arms, dtype=float)
                self.arm_samples = np.zeros(self.n_arms, dtype=int)
                for a in self.arms:
                    self.resat_arms[a].append(self.t)
            
            else: # Reset only the arm that triggered the alarm
                self.means[arm_index] = 0.0
                self.arm_samples[arm_index] =0
                self.resat_arms[arm].append(self.t)
            
    def reset(self) -> None:
        super().reset()
        self.resat_arms = {arm : [] for arm in self.arms}