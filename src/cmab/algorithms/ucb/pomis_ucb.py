from cmab.algorithms.base import BaseBanditAlgorithm
import numpy as np

from cmab.algorithms.base import BaseBanditAlgorithm

from typing import Set, List, Tuple, FrozenSet, AbstractSet
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
        self.actions = POMISs(G, Y)
        self.n_arms = len(self.actions)
        self.estimates = np.zeros(self.n_arms)
        self.action_samples = np.zeros(self.n_arms)
        self.t = 0
        print(f"POMISs found: {self.actions}")

    def select_arm(self):
        for i in range(self.n_arms):   # ensure each arm is tried once
            if self.action_samples[i] == 0:
                return i

        ucb_values = []
        for action in range(self.n_arms): 
            n_i = self.action_samples[action]
            bound = np.sqrt(np.log(self.t)/n_i)
            ucb_values.append(self.estimates[action] + self.c*bound)
        return np.argmax(ucb_values)
    
    def _update(self, action, reward):
        self.t += 1
        self.action_samples[action] += 1
        num_samples = self.action_samples[action]
        prev_reward = self.estimates[action]
        self.estimates[action] = prev_reward + 1/(num_samples)*(reward - prev_reward)
    
    def reset(self):
        self.t = 0
        self.estimates = np.zeros(self.n_arms)
        self.action_samples = np.zeros(self.n_arms)
