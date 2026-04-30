from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import Intervention
from .base import BaseCausalBanditEnv
from cmab.typing import MechanismChangeEvent, ShiftEvent
    

class NSCausalBanditEnv(BaseCausalBanditEnv):
    def __init__(self, scm: SCM, 
                 reward_node: str, 
                 side_observations: bool = True, 
                 seed:int=42, atomic: bool = False,
                 non_intervenable: list[str] = [], 
                 include_empty: bool = True, 
                 change_variables: list[str] = [], 
                 updates: list[str | float] = [], 
                 change_points: list[int] = [],
                 ):
        super().__init__(scm, reward_node, side_observations, seed, atomic, non_intervenable, include_empty=include_empty)
        self.change_variables = change_variables
        self.updates = updates
        self.change_points = change_points
        self._idx = 0  # index to keep track of which variable to change next

    def _change_point(self):
        variable= self.change_variables[self._idx]
        update = self.updates[self._idx]
        if isinstance(update, str):
            event = MechanismChangeEvent(variable=variable, new_mechanism=update)
        else:             
            event = ShiftEvent(variable=variable, new_param={"p": update})
        self._idx += 1
        self.scm.apply_change_event(event)

    def get_change_points(self)-> list[int]:
        return self.change_points

    def step(self, action: Intervention):

        if self._step in self.change_points and self._idx < len(self.change_variables):
            self._change_point()

        self._step += 1
        values = self.scm.sample(intervention=action)
        
        if self.side_observations:
            return self._get_obs(), values, False, False, self._get_info()  # observation, reward, terminated, truncated, info
        
        return self._get_obs(), values[self.reward_node], False, False, self._get_info()  # observation, reward, terminated, truncated, info

    def reset(self, scm_seed:int = None, ns_seed:int = None):
        """Seed used to reset the SCM
        ns_seed used to reset the non-stationarity rng"""
        self._step = 0
        self._idx = 0  

        if scm_seed is not None:
            self.scm.reset(seed=scm_seed)
        
        if ns_seed is not None:
            self.seed = ns_seed
            self.rng = np.random.default_rng(seed=ns_seed)