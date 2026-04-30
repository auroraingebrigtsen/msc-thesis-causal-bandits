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
        
        variable= self.variables[self._idx]
        update = self.update[self._idx]
        if isinstance(update, str):
            event = MechanismChangeEvent(variable=variable, new_mechanism=update)
        else:             
            event = ShiftEvent(variable=variable, new_param={"p": update})
        self._idx += 1
        return event
    
    def get_change_points(self, T:int) -> list[int]:
        """ Returns an array of T entries with 0 at time steps where no change occurs and 1 at time steps where a change occurs. """
        return [t for t in range(1, T) if t % self.every == 0 and (t // self.every) <= len(self.variables)]
    
    def reset(self) -> None:
        self._idx = 0
