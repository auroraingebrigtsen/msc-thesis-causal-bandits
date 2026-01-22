from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
import numpy as np
from typing import override

class MyFirstAgent(PomisUCBAgent):
    def __init__(self, G, Y, c:float=2):
        super().__init__(G, Y, c)
        self.cpds = {}

    @override
    def _update(self, arm_index: int, reward: float) -> None:
        super()._update(arm_index, reward)
        for arm in range(self.n_arms):
            if self._is_change_point(arm):
                self._update_on_change_point()


    def _is_change_point(self, arm: int) -> bool:
        return False

    def _update_on_change_point(self):
        