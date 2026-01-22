from cmab.algorithms.base import BaseBanditAlgorithm
import numpy as np

from cmab.algorithms.base import BaseBanditAlgorithm

from cmab.scm.scm import SCM
from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs

class PomisUCBAgent(BaseBanditAlgorithm):
    """
    Args:
    c: float, degree of exploration
    """
    def __init__(self, G: CausalDiagram, Y: str, c:float=2):
        self.G = G
        self.Y = Y
        self.c = c
        self.arms = list(POMISs(G, Y))
        self.n_arms = len(self.arms)
        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
        self.t = 0
        print(f"POMISs found: {self.arms}")

    def select_arm(self):
        for i in range(self.n_arms):   # ensure each arm is tried once
            if self.arm_samples[i] == 0:
                return self.arms[i]

        ucb_values = []
        for arm in range(self.n_arms): 
            n_i = self.arm_samples[arm]
            bound = np.sqrt(np.log(self.t)/n_i)
            ucb_values.append(self.estimates[arm] + self.c*bound)
        return self.arms[np.argmax(ucb_values)]
    
    def _update(self, arm, reward):
        self.t += 1
        self.arm_samples[arm] += 1
        num_samples = self.arm_samples[arm]
        prev_reward = self.estimates[arm]
        self.estimates[arm] = prev_reward + 1/(num_samples)*(reward - prev_reward)
    
    def reset(self):
        self.t = 0
        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
