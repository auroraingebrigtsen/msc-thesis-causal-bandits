from cmab.algorithms.ucb.ucb_base import UCBAgent
from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs
from cmab.typing import Intervention
import numpy as np

class PomisUCBAgent(UCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[Intervention], c: float=np.sqrt(2), atomic:bool=False):
        self.G = G
        self.arms = self._get_pomis_arms(all_arms=arms, reward_node=reward_node, atomic=atomic) 
        self.atomic = atomic
        super().__init__(reward_node, self.arms, c)
        print(f"Total: {len(self.arms)} arms selected from {len(arms)} total arms.")
        print("Pomis arms:", self.arms)

    def _get_pomis_arms(self, all_arms: list[Intervention], reward_node: str, atomic: bool) -> list[Intervention]:
        """Select only arms that correspond to POMISs."""
        if atomic:
            return all_arms
        pomis_sets = set(POMISs(self.G, reward_node))
        return [arm for arm in all_arms if frozenset(var for var, _ in arm) in pomis_sets]