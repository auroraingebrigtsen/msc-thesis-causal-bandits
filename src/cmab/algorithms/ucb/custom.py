from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.algorithms.ucb.ucb_base import UCBAgent
from typing import override
from cmab.algorithms.cpd.ph_cpd import PageHinkleyCPD
from cmab.scm .causal_diagram import CausalDiagram
from cmab.typing import InterventionSet, Observation


class MyFirstAgent(PomisUCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float=2):
        super().__init__(reward_node, G, arms, c)
        self.nodes = list(G.nodes - {reward_node})

        self.cpds = {node: PageHinkleyCPD(node, G.Pa({node}, include_self=False)) for node in self.nodes}

    @override
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        super()._update(arm, observation)
        detected = set()
        for node in self.nodes:
            if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
                continue
            cpd = self.cpds[node]
            change_point = cpd.update(observation)

            if change_point:
                print(f"Change point detected at node {node}!")
                detected.add(node)
        
        if detected:
            self._update_on_change_point(detected)

    def _update_on_change_point(self, detected: set[str]) -> None:

        # We know which variable was alarmed, it must be the exogenous variabe
        # of this one that shifted. 
        shifted = set()
        for node in detected:
            exogenous = {u for (u, v) in self.G.noise_vars if v == node}
            shifted.update(exogenous)
        # TODO: In this first situation we assume markovianity. Extend, but need to think more how this will work

        affected_vars = set()
        for node in self.nodes:
            sub = self.G.do(intervention_set={node})
            for shifted_u in shifted:
                if not sub.d_separated({node}, {shifted_u}, set()):
                    affected_vars.add(node)
        
        for node in affected_vars:
            arm_index = self.arm_to_index.get(node)  # TODO fix so it updates all sets involving node
            self.estimates[arm_index] = 0.0
            self.arm_samples[arm_index] = 0
            self.cpds[node].reset()
















class MyFirstAtomicAgent(UCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float=2):
        super().__init__(reward_node, arms, c)
        self.G = G
        self.nodes = list(G.nodes - {reward_node})

        self.cpds = {node: PageHinkleyCPD(node, G.Pa({node}, include_self=False)) for node in self.nodes}

    @override
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        super()._update(arm, observation)
        detected = set()
        for node in self.nodes:
            if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
                continue
            cpd = self.cpds[node]
            change_point = cpd.update(observation)

            if change_point:
                print(f"Change point detected at node {node}!")
                detected.add(node)
        
        if detected:
            self._update_on_change_point(detected)

    def _update_on_change_point(self, detected: set[str]) -> None:

        # We know which variable was alarmed, it must be the exogenous variabe
        # of this one that shifted. 
        shifted = set()
        for node in detected:
            exogenous = {u for (u, v) in self.G.noise_vars if v == node}
            shifted.update(exogenous)
        # TODO: In this first situation we assume markovianity. Extend, but need to think more how this will work

        affected_vars = set()
        for node in self.nodes:
            sub = self.G.do(intervention_set={node})
            for shifted_u in shifted:
                if not sub.d_separated({node}, {shifted_u}, set()):
                    affected_vars.add(node)
        
        for node in affected_vars:
            arm_index = self.arms.index({(node, None)})  # atomic interventions
            self.estimates[arm_index] = 0.0
            self.arm_samples[arm_index] = 0
            self.cpds[node].reset()

    @override
    def reset(self):
        super().reset()
        for cpd in self.cpds.values():
            cpd.reset()