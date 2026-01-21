from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseMechanism(ABC):
    def __init__(self, v_parents: list[str], u_parents: list[str]):
        self.v_parents = v_parents
        self.u_parents = u_parents

    @abstractmethod
    def __call__(self, v_vals: dict[str, Any], u_vals: dict[str, Any]) -> Any:
        pass
