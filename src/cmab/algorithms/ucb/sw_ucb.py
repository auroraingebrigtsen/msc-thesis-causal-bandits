# Sliding Window UCB Algorithm
import numpy as np
from collections import deque
from cmab.algorithms.base import BaseBanditAlgorithm
from cmab.typing import Intervention, Observation

class SlidingWindowUCBAgent(BaseBanditAlgorithm):
    def __init__(self, reward_node:str, arms, c:float=2.0, window_size:int=50):
        super().__init__(reward_node)
        self.arms = arms
        self.n_arms = len(arms)
        self.c = c
        self.means=np.zeros(self.n_arms)
        self.W=window_size
        self.t=0
        self.buffers=[deque(maxlen=self.W) for _ in range(self.n_arms)]

    def select_arm(self) -> Intervention:
        for i in range(self.n_arms):   # ensure each arm is tried once
            if len(self.buffers[i]) == 0:
                return self.arms[i]

        L = min(max(1, self.t), self.W) 
        ucb_values = []
        for i in range(self.n_arms):
            n_i = len(self.buffers[i])
            bound = self.c * np.sqrt(np.log(L) / n_i)
            ucb_values.append(self.means[i] + bound)
        return self.arms[np.argmax(ucb_values)]

    def _update(self, arm: Intervention, observation: Observation) -> None:
        reward = observation[self.reward_node]
        self.t += 1
        arm_index = self.arm_to_index[arm]
        buf = self.buffers[arm_index]
        buf.append(reward)
        self.means[arm_index] = sum(buf) / len(buf)

    def reset(self):
        self.t=0
        self.buffers=[deque(maxlen=self.W) for _ in range(self.n_arms)]
        self.means = np.zeros(self.n_arms)