from cmab.algorithms.base import BaseBanditAlgorithm

from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs

class PomisKLUCBAgent(BaseBanditAlgorithm):
    """ TODO: POMIS based KL-UCB """

    def __init__(self, G: CausalDiagram, Y: str, f, T):
        self.G = G
        self.Y = Y
        self.actions = POMISs(G, Y)
        self.n_arms = len(self.actions)

    def select_arm(self) -> int:
        pass 

    def _update(self, arm_index: int, reward: float) -> None:
        pass

    def reset(self):
        return super().reset()