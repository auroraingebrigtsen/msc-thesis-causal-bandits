from dataclasses import dataclass
from typing import Any, Tuple, Dict, TypeAlias, Callable

Assignment: TypeAlias = Tuple[str, float]
Intervention: TypeAlias = frozenset[Assignment]

Observation: TypeAlias = Dict[str, float]


@dataclass(frozen=True)
class BaseEvent:
    variable: str

@dataclass(frozen=True)
class ShiftEvent(BaseEvent):
    new_param: dict[str, float] # e.g. ["p" : 0.7] 

@dataclass(frozen=True)
class MechanismChangeEvent(BaseEvent):
    new_mechanism: str # e.g."int(u['U_X'])"

    def as_callable(self) -> Callable:
        return eval(self.new_mechanism)
    
@dataclass(frozen=True)
class LinearMechanismChangeEvent(BaseEvent):
    new_weights: dict[str, float] 