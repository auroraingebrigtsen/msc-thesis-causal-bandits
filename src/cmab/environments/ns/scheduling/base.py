from abc import ABC, abstractmethod
from typing import Optional
from cmab.scm.scm import SCM
from cmab.typing import ShiftEvent
import numpy as np

class ShiftSchedule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def next(self, t: int, scm: SCM, rng) -> Optional[ShiftEvent]:
        pass

    @abstractmethod
    def get_change_points(self, T: int, rng: np.random.Generator = None) -> list[int]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass