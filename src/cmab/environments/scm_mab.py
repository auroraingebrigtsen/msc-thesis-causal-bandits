from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import InterventionSet, Intervention
import itertools
from .base import BaseCausalBanditEnv

class  CausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm: SCM, reward_node: str, side_observations: bool = True, seed=42, atomic: bool = False, non_intervenable: list[str] = []):
        super().__init__(scm, reward_node, side_observations, seed, atomic, non_intervenable)

    def step(self, action: InterventionSet):
        self._step += 1
        values = self.scm.sample(intervention_set=action)
        
        if self.side_observations:
            return self._get_obs(), values, False, False, self._get_info()  # observation, reward, terminated, truncated, info
        
        return self._get_obs(), values[self.reward_node], False, False, self._get_info()  # observation, reward, terminated, truncated, info

    def reset(self, seed: int = None):
        self._step = 0
        self.state = None
        self.scm.reset(seed=seed)