from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
import numpy as np
from typing import override
from cmab.algorithms.cpd.ph_cpd import PageHinkleyCPD
from cmab.scm .causal_diagram import CausalDiagram
from cmab.typing import InterventionSet, Observation


class MyFirstAgent(PomisUCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, c:float=2):
        super().__init__(G, reward_node, c)
        self.nodes = list(G.V)

        self.cpds = {node: PageHinkleyCPD(node, G.Pa(node, include_self=False)) for node in self.nodes}

    @override
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        super()._update(arm, observation)
        detected = set()
        for node in self.nodes:
            if node in arm:  # Dont update cpd for intervened nodes
                continue
            cpd = self.cpds[node]
            change_point = cpd.update(observation)

            if change_point:
                print(f"Change point detected at node {node}!")
                detected.add(node)
        
        self._update_on_change_point(detected)

    def _update_on_change_point(self, detected: set[str]) -> None:
        non_affected_nodes = []
        for node in detected:
            sub = self.G.do(detected)
            