from abc import ABC, abstractmethod
from typing import Optional
from cmab.scm.scm import SCM
from cmab.typing import ShiftEvent
import numpy as np

class BaseSchedule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def next(self, t: int) -> Optional[ShiftEvent]:
        pass

    @abstractmethod
    def get_change_points(self, T: int, rng: np.random.Generator = None) -> list[int]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
