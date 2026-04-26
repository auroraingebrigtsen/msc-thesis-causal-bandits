from .base import BaseSchedule
from cmab.typing import ShiftEvent, MechanismChangeEvent
from typing import Optional

class ControlledMechanismChangeSchedule(BaseSchedule):
    def __init__(self, variables: list[str], new_mechanisms: list[str], every: int):
        assert len(variables) == len(new_mechanisms)
        self.variables = variables
        self.new_mechanisms = new_mechanisms
        self.every = every  # Apply a shift at every t steps
        self._idx = 0
       
    def next(self, t: int) -> Optional[MechanismChangeEvent]:
        if t == 0 or (t % self.every != 0):
            return None
        
        if self._idx >= len(self.variables):
            return None
        
        event = MechanismChangeEvent(variable=self.variables[self._idx], new_mechanism=self.new_mechanisms[self._idx])
        self._idx += 1
        return event
    
    def get_change_points(self, T:int) -> list[int]:
        return [t for t in range(1, T) if t % self.every == 0]
    
    def reset(self) -> None:
        self._idx = 0

class ControlledShiftSchedule(BaseSchedule):
    def __init__(self, variables: list[str], new_params: list[float], every: int):
        assert len(variables) == len(new_params)
        self.variables = list(variables)
        self.new_params = list(new_params)
        self.every = every  # Apply a shift at every t steps
        self._idx = 0
       
    def next(self, t: int) -> Optional[ShiftEvent]:
        if t == 0 or (t % self.every != 0):
            return None
        
        if self._idx >= len(self.variables):
            return None
        
        event = ShiftEvent(variable=self.variables[self._idx], new_param={"p": self.new_params[self._idx]})
        self._idx += 1
        return event
    
    def get_change_points(self, T:int) -> list[int]:
        return [t for t in range(1, T) if t % self.every == 0]
    
    def reset(self) -> None:
        self._idx = 0
