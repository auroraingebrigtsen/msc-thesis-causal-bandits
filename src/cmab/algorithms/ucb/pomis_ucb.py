import numpy as np
from cmab.algorithms.base import BaseBanditAlgorithm
from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs
from cmab.typing import InterventionSet, Observation
from collections import defaultdict

class PomisUCBAgent(BaseBanditAlgorithm):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c: float = 2):
        super().__init__(reward_node)
        self.G = G
        self.all_arms = arms
        self.c = c

        self.arms = self._get_arms_from_pomis_sets()
        print(f"Selected arms from POMISs: {self.arms}")
        print(f"Total: {len(self.arms)}")
        self.n_arms = len(self.arms)

        self.arm_to_index = {arm: i for i, arm in enumerate(self.arms)}

        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
        self.t = 0

    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        reward = observation[self.reward_node]
        self.t += 1
        arm_index = self.arm_to_index[arm]
        self.arm_samples[arm_index] += 1
        n = self.arm_samples[arm_index]
        prev = self.estimates[arm_index]
        self.estimates[arm_index] = prev + (reward - prev) / n


    def _get_arms_from_pomis_sets(self) -> list[InterventionSet]:
        """Select arms that correspond to POMISs."""
        pomis_sets = set(POMISs(self.G, self.reward_node))  # membership test

        return [
            arm
            for arm in self.all_arms
            if frozenset(var for var, _ in arm) in pomis_sets
        ]

    def select_arm(self) -> InterventionSet:
        for i in range(self.n_arms):   # ensure each arm is tried once
            if self.arm_samples[i] == 0:
                return self.arms[i]

        ucb_values = []
        for arm in range(self.n_arms): 
            n_i = self.arm_samples[arm]
            bound = np.sqrt(np.log(self.t)/n_i)
            ucb_values.append(self.estimates[arm] + self.c*bound)
        return self.arms[np.argmax(ucb_values)]
    
    
    def reset(self):
        self.t = 0
        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
