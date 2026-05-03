from cmab.algorithms.base import BaseBanditAlgorithm
import numpy as np
from cmab.typing import Intervention, Observation

class UCBAgent(BaseBanditAlgorithm):
    """
    Args:
    c: float, degree of exploration
    """
    def __init__(self, reward_node:str, arms: list[Intervention], c:float=np.sqrt(2)):
        super().__init__(reward_node)
        self.arms = arms
        self.n_arms = len(arms)
        self.c = c
        self.means = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.t = 0
        self.arm_to_index = {arm: idx for idx, arm in enumerate(arms)}

    def select_arm(self) -> Intervention:
        for i in range(self.n_arms):   # ensure each arm is tried once
            if self.arm_samples[i] == 0:
                return self.arms[i]

        ucb_values = []
        for arm in range(self.n_arms): 
            n_arm = self.arm_samples[arm]
            t = np.sum(self.arm_samples)  
            bound = np.sqrt(np.log(t)/n_arm)
            ucb_values.append(self.means[arm] + self.c*bound)
        return self.arms[np.argmax(ucb_values)]
    
    def _update(self, arm: Intervention, observation: Observation) -> None:
        reward = observation[self.reward_node]
        arm_index = self.arm_to_index[arm]
        self.arm_samples[arm_index] += 1
        num_samples = self.arm_samples[arm_index]
        prev_mean = self.means[arm_index]
        self.means[arm_index] = prev_mean + 1/(num_samples)*(reward - prev_mean)
        self.t += 1
    
    def reset(self):
        self.t = 0
        self.means = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
