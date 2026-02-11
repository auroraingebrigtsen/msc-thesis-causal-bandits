from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import InterventionSet
from .base import BaseCausalBanditEnv
    

class NSCausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm: SCM, reward_node: str, side_observations: bool = True, seed:int=42, atomic: bool = False, non_intervenable: list[str] = [], prob_distribution_shift: float = 0.01, max_delta: float = 0.2, 
                 exogenous_to_shift: list[str] = [], new_probs: list[float] = []):
        super().__init__(scm, reward_node, side_observations, seed, atomic, non_intervenable)
        self.prob_distribution_shift = prob_distribution_shift
        self.state = None
        self.max_delta = max_delta
        self.action_space: list[InterventionSet] = self._init_action_space(atomic=atomic, non_intervenable=non_intervenable)
        self.exogenous_to_shift_init = exogenous_to_shift
        self.new_probs_init = new_probs
        self.exogenous_to_shift = exogenous_to_shift.copy()
        self.new_probs = new_probs.copy()
        self.controlled_shifts = bool(exogenous_to_shift and new_probs)


    def step(self, action: InterventionSet):
        # Possibly apply a distribution shift
        # if self.rng.uniform(0, 1) < self.prob_distribution_shift:
        #         self.scm.exogenous_distribution_shift(max_delta=self.max_delta, rng=self.rng)
        if self._step % 200 == 0:
            print(f"EXogenous left to shift: {self.exogenous_to_shift}")
            print (f"\nStep {self._step}: Applying distribution shift.")
            if self.controlled_shifts:
                exogenous_to_shift = self.exogenous_to_shift.pop(0)
                new_prob = self.new_probs.pop(0)
            self.scm.exogenous_distribution_shift(max_delta=self.max_delta, rng=self.rng, exogenous_to_shift=exogenous_to_shift, new_prob=new_prob)

        self._step += 1
        values = self.scm.sample(intervention_set=action)
        
        if self.side_observations:
            return self._get_obs(), values, False, False, self._get_info()  # observation, reward, terminated, truncated, info
        
        return self._get_obs(), values[self.reward_node], False, False, self._get_info()  # observation, reward, terminated, truncated, info

    def reset(self, scm_seed:int = None, ns_seed:int = None):
        """Seed used to reset the SCM
        ns_seed used to reset the non-stationarity rng"""
        self._step = 0
        self.state = None

        if scm_seed is not None:
            self.scm.reset(seed=scm_seed)
        
        if ns_seed is not None:
            self.seed = ns_seed
            self.rng = np.random.default_rng(seed=ns_seed)
            self.exogenous_to_shift = self.exogenous_to_shift_init.copy()
            self.new_probs = self.new_probs_init.copy()