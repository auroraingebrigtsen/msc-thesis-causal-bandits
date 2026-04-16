from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import Intervention
from .base import BaseCausalBanditEnv
from .ns.scheduling.base import BaseSchedule
from cmab.typing import ShiftEvent, MechanismChangeEvent
    

class NSCausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm: SCM, reward_node: str, side_observations: bool = True, seed:int=42, atomic: bool = False,
                  non_intervenable: list[str] = [], schedule: BaseSchedule = None, include_empty: bool = True):
        super().__init__(scm, reward_node, side_observations, seed, atomic, non_intervenable, include_empty=include_empty)
        self.state = None
        self.schedule = schedule

    def step(self, action: Intervention):

        if self.schedule is not None:
            event = self.schedule.next(t=self._step)

            if isinstance(event, ShiftEvent):
                self.scm.apply_shift(event)
            elif isinstance(event, MechanismChangeEvent):
                self.scm.apply_mechanism_change(event)
            
        self._step += 1
        values = self.scm.sample(intervention=action)
        
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
        
        if self.schedule is not None:
            self.schedule.reset()