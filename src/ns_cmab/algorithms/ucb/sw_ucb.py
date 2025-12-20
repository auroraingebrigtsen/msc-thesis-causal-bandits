# Sliding Window UCB Algorithm
import numpy as np
import sys
import os
from collections import deque

from base import Base

class UCBAgent(Base):
    """
    Args:
    c: float, degree of exploration
    """
    def __init__(self, bandits, c:float=2, random_state: int=42):
        super().__init__(bandits, random_state)
        self.c = c
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)

    def _update(self, action, reward):
        self.action_samples[action] += 1
        num_samples = self.action_samples[action]
        prev_reward = self.estimates[action]
        self.estimates[action] = prev_reward + 1/(num_samples)*(reward - prev_reward)
    
    def select_action(self):
        ucb_estimates = []
        total_runs = np.sum(self.action_samples) + 1 # starting with first run
        for action in range(self.k): 
            action_samples = self.action_samples[action] if self.action_samples[action] != 0 else 1
            bound = np.sqrt(np.log(total_runs)/action_samples)
            ucb_estimates.append(self.estimates[action] + self.c*bound)
        return np.argmax(ucb_estimates)
    
    def reset(self):
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)
        self.regret = []
        self.averaged_regret = []
        self.random_state += 1
        np.random.seed(self.random_state)


class SlidingWindowUCBAgent(BaseAgent):
    def __init__(self, bandits, c=1.0, window_size=50, random_state=42):
        super().__init__(bandits, random_state)
        self.c           = c
        self.W           = window_size
        self.t           = 0
        # one deque per arm, maxlen=W
        self.buffers     = [deque(maxlen=self.W) for _ in range(self.k)]
        # empirical means
        self.estimates   = np.zeros(self.k)

    def select_action(self):
        # 1) ensure each arm is tried once
        for i in range(self.k):
            if len(self.buffers[i]) == 0:
                return i

        # 2) compute UCB on the sliding window
        L = min(max(1, self.t), self.W)
        ucb_vals = []
        for i in range(self.k):
            n_i = len(self.buffers[i])
            bonus = self.c * np.sqrt(np.log(L) / n_i)
            ucb_vals.append(self.estimates[i] + bonus)

        return int(np.argmax(ucb_vals))

    def _update(self, action, reward):
        self.t += 1
        buf = self.buffers[action]
        buf.append(reward)
        self.estimates[action] = sum(buf) / len(buf)

    def reset(self):
        self.t         = 0
        self.buffers   = [deque(maxlen=self.W) for _ in range(self.k)]
        self.estimates = np.zeros(self.k)
        # if BaseAgent tracks regret etc., you may want to reset those too
        self.random_state += 1
        np.random.seed(self.random_state)
        self.regret = []
        self.averaged_regret = []

class PageHinkleyUCBAgent(BaseAgent):
    """
    UCB agent with a Page–Hinkley change detector per arm (no sigma required).

    Args:
        c:       float, exploration coefficient
        delta:   float, minimum change to detect (drift allowance)
        h:       float, detection threshold
    """
    def __init__(self, bandits, c: float = 2.0,
                 delta: float = 0.5, h: float = 5.0,
                 random_state: int = 42):
        super().__init__(bandits, random_state)
        self.c = c
        self.delta = delta
        self.h = h
        # UCB statistics
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)
        # Page–Hinkley statistics
        self.ph_stat = np.zeros(self.k)
        self.baselines = np.zeros(self.k)

    def _update(self, action, reward):
        b_old = self.baselines[action]
        # compute Page–Hinkley increment
        diff = reward - b_old - self.delta
        S_new = max(0.0, self.ph_stat[action] + diff)

        if S_new > self.h:
            self.ph_stat[action] = 0.0
            self.action_samples[action] = 1
            self.estimates[action] = reward
            self.baselines[action] = reward
        else:
            self.ph_stat[action] = S_new
            n_old = self.action_samples[action]
            n = n_old + 1
            self.baselines[action] = b_old + (reward - b_old) / n
            prev = self.estimates[action]
            self.estimates[action] = prev + (reward - prev) / n
            self.action_samples[action] = n

    def select_action(self):
        ucb_estimates = []
        total = np.sum(self.action_samples) + 1  # avoid log(0)
        for i in range(self.k):
            n_i = self.action_samples[i] if self.action_samples[i] > 0 else 1
            bonus = np.sqrt(np.log(total) / n_i)
            ucb_estimates.append(self.estimates[i] + self.c * bonus)
        return int(np.argmax(ucb_estimates))

    def reset(self):
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)
        self.ph_stat = np.zeros(self.k)
        self.baselines = np.zeros(self.k)
        self.regret = []
        self.averaged_regret = []
        self.random_state += 1
        np.random.seed(self.random_state)