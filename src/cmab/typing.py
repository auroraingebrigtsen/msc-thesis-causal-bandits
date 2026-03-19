from dataclasses import dataclass
from typing import Tuple, Dict, TypeAlias

Assignment: TypeAlias = Tuple[str, float]
Intervention: TypeAlias = frozenset[Assignment]

Observation: TypeAlias = Dict[str, float]

@dataclass(frozen=True)
class ShiftEvent:
    exogenous: str
    param_updates: dict[str, float] # e.g. ["p" : 0.7] for a binary variable, or {"mean": 0.5, "std": 0.1} for a Gaussian variable