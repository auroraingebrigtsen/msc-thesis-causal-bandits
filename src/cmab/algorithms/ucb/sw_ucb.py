# Sliding Window UCB Algorithm
import numpy as np
from collections import deque
from cmab.algorithms.base import BaseBanditAlgorithm

class SlidingWindowUCBAgent(BaseBanditAlgorithm):
    def __init__(self,n_arms, c:float=2.0, window_size:int=50):
        self.n_arms = n_arms
        self.c =c
        self.estimates=np.zeros(self.n_arms)
        self.W=window_size
        self.t=0
        self.buffers=[deque(maxlen=self.W) for _ in range(self.n_arms)]

    def select_arm(self):
        for i in range(self.n_arms):   # ensure each arm is tried once
            if len(self.buffers[i]) == 0:
                return i

        L = min(max(1, self.t), self.W) 
        ucb_values = []
        for i in range(self.n_arms):
            n_i = len(self.buffers[i])
            bound = self.c * np.sqrt(np.log(L) / n_i)
            ucb_values.append(self.estimates[i] + bound)
        return int(np.argmax(ucb_values))

    def _update(self, arm, reward):
        self.t += 1
        buf = self.buffers[arm]
        buf.append(reward)
        self.estimates[arm] = sum(buf) / len(buf)

    def reset(self):
        self.t=0
        self.buffers=[deque(maxlen=self.W) for _ in range(self.n_arms)]
        self.estimates = np.zeros(self.n_arms)