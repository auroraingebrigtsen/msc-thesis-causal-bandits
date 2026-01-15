from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseMechanism(ABC):
    def __init__(self, parents: list[str], u_parents: list[str], seed: int):
        self.parents = parents
        self.u_parents = u_parents
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def __call__(self, v_vals: dict[str, Any], u_vals: dict[str, Any]) -> Any:
        pass
