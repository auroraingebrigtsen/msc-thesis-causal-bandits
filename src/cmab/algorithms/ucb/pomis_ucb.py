from cmab.algorithms.ucb.ucb_base import UCBAgent
from cmab.scm.causal_diagram import CausalDiagram
from cmab.algorithms.pomis.pomis_sets import POMISs
from cmab.typing import InterventionSet

class PomisUCBAgent(UCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c: float = 2):
        self.G = G
        self.arms = self._get_arms_from_pomis_sets(arms, reward_node)
        super().__init__(reward_node, self.arms, c)
        print(f"Selected arms from POMISs: {self.arms}")
        print(f"Total: {len(self.arms)} arms selected from {len(arms)} total arms.")

    def _get_arms_from_pomis_sets(self, all_arms, reward_node) -> list[InterventionSet]:
        """Select arms that correspond to POMISs."""
        pomis_sets = set(POMISs(self.G, reward_node))

        return [frozenset()] + [
            arm
            for arm in all_arms
            if frozenset(var for var, _ in arm) in pomis_sets
        ]