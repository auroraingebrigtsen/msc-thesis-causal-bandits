from .base import ShiftSchedule
from cmab.typing import ShiftEvent
from cmab.scm.scm import SCM
import numpy as np
from typing import Optional

class ControlledSchedule(ShiftSchedule):
    def __init__(self, exogenous: list[str], new_params: list[float], every: int):
        assert len(exogenous) == len(new_params)
        self.exogenous = list(exogenous)
        self.new_params = list(new_params)
        self.every = every  # Apply a shift at every t steps
        self._idx = 0
       
    def next(self, t: int, scm: SCM, rng: np.random.Generator) -> Optional[ShiftEvent]:
        if t == 0 or (t % self.every != 0):
            return None
        
        if self._idx >= len(self.exogenous):
            return None
        
        event = ShiftEvent(exogenous=self.exogenous[self._idx], param_updates={"p": self.new_params[self._idx]})
        self._idx += 1
        return event
    
    def reset(self) -> None:
        self._idx = 0
