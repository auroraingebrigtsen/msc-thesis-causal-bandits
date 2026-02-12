from .base import ShiftSchedule
from cmab.typing import ShiftEvent
from cmab.scm.scm import SCM
import numpy as np
from typing import Optional

class RandomSchedule(ShiftSchedule):
    def __init__(self, prob_shift: float, max_delta: float):
        self.prob_shift = prob_shift
        self.max_delta = max_delta

    def next(self, t: int, scm: SCM, rng: np.random.Generator) -> Optional[ShiftEvent]:
        if rng.uniform(0, 1) >= self.prob_shift:
            return None
        
        u = rng.choice(scm.U)
        dist = scm.P_u_marginals[u]

        delta = rng.uniform(-self.max_delta, self.max_delta)
        new_p = min(1.0, max(0.0, dist.p + delta))
        return ShiftEvent(exogenous=u, param_updates={"p": new_p})

    def reset(self) -> None:
        pass