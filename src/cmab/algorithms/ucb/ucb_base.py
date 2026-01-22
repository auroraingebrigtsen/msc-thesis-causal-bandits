from cmab.algorithms.base import BaseBanditAlgorithm
import numpy as np

class UCBAgent(BaseBanditAlgorithm):
    """
    Args:
    c: float, degree of exploration
    """
    def __init__(self, arms, c:float=2):
        self.arms = arms
        self.n_arms = len(arms)
        self.c = c
        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
        self.t = 0

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
        arm_index = self.arms.index(arm)
        self.arm_samples[arm_index] += 1
        num_samples = self.arm_samples[arm_index]
        prev_reward = self.estimates[arm_index]
        self.estimates[arm_index] = prev_reward + 1/(num_samples)*(reward - prev_reward)
    
    def reset(self):
        self.t = 0
        self.estimates = np.zeros(self.n_arms)
        self.arm_samples = np.zeros(self.n_arms)
