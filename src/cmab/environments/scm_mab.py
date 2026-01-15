from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import InterventionSet, Intervention
import itertools
from .base import BaseCausalBanditEnv

class  CausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm, reward_node, seed = 42):
        super().__init__(scm, reward_node, seed)

    def step(self, action: InterventionSet):
        self._step += 1
        reward = self.scm.sample(intervention_set=action)[self.reward_node]
        return self._get_obs(), reward, False, False, self._get_info()  # observation, reward, terminated, truncated, info

    def reset(self):
        self._step = 0
        self.state = None
        self.scm.reset()