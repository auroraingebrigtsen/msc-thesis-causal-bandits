from .base import BaseSchedule
from cmab.typing import ShiftEvent, MechanismChangeEvent
from typing import Optional

class ControlledSchedule(BaseSchedule):
    def __init__(self, variables: list[str], update: list[str | float], every: int):
        """A schedule that applies either a mechanism change or a shift at fixed intervals.
        Args:
            variables (list[str]): The variables to be changed.
            update (list[str | float]): The new mechanisms or parameters for the changes. 
                If an element is a string, it is treated as a new mechanism. If it is a float, it is treated as a new parameter for a shift.
            every (int): The interval at which the changes are applied.
        """
        assert len(variables) == len(update)
        self.variables = variables
        self.update = update
        self.every = every  # Apply a shift at every t steps
        self._idx = 0
       
    def next(self, t: int) -> Optional[MechanismChangeEvent]:
        if t == 0 or (t % self.every != 0) or self._idx >= len(self.variables):
            return None
        
        if isinstance(self.update[self._idx], str):
            event = MechanismChangeEvent(variable=self.variables[self._idx], new_mechanism=self.update[self._idx])
        else:             
            event = ShiftEvent(variable=self.variables[self._idx], new_param={"p": self.update[self._idx]})
        self._idx += 1
        return event
    
    def get_change_points(self, T:int) -> list[int]:
        return [t for t in range(1, T) if t % self.every == 0]
    
    def reset(self) -> None:
        self._idx = 0
