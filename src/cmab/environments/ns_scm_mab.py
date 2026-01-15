from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import InterventionSet
from .base import BaseCausalBanditEnv

class NSCausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm: SCM, reward_node: str, prob_distribution_shift: float = 0.01, max_delta: float = 0.2, seed:int=42):
        self.scm = scm
        self.reward_node = reward_node
        self._step = 0
        self.prob_distribution_shift = prob_distribution_shift
        self.rng = np.random.default_rng(seed=seed)
        self.state = None
        self.max_delta = max_delta
        self.action_space: list[InterventionSet] = self._init_action_space()

    def step(self, action: InterventionSet):

        # Possibly apply a distribution shift
        if self.rng.uniform(0, 1) < self.prob_distribution_shift:
                self.scm.exogenous_distribution_shift(max_delta=self.max_delta)

        self._step += 1
        reward = self.scm.sample(intervention_set=action)[self.reward_node]
        return self._get_obs(), reward, False, False, self._get_info()  # observation, reward, terminated, truncated, info

    def reset(self):
        self._step = 0
        self.state = None
        self.scm.reset()
        # must decide wheter to reset seed, will produce different change points in each reset