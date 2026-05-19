from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
from river import drift
import numpy as np
from cmab.algorithms.pomis.pomis_sets import MUCT

class PHTSrUCBAgent(PomisUCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[Intervention], c:float=np.sqrt(2), atomic:bool=False, 
                 delta:float=0.5, lambda_:float=5.0, min_samples_for_detection:int=10):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.G = G
        self.nodes = list(G.nodes)
        self.parents = {node: list(G.Pa({node}, include_self=False)) for node in self.nodes}
        
        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        self.cpds = {node: {} for node in self.nodes} # cpds[node][parent_cfg] gives the drift detector for that node and parent configuration
        self.resat_arms = {arm: [] for arm in self.arms}  # Keep track of detected change points for analysis 
        self.detected_nodes = {node : [] for node in self.nodes} # Keep track of detected change points for each node for analysis

    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        detected = set()
        for node in self.nodes:
            if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
                continue

            cfg = tuple(observation[parent] for parent in self.parents[node])
            if cfg not in self.cpds[node]:
                self.cpds[node][cfg] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
            cpd = self.cpds[node][cfg]
            cpd.update(observation[node])

            if self.cpds[node][cfg].drift_detected:
                print(f"\nStep {self.t}: Change point detected for node {node}!")
                print(f"Parent configuration: {cfg}")
                detected.add(node)
                self.detected_nodes[node].append(self.t)

        if len(detected) > 0:
            # Add variables sharing an UC with nodes in detected, to detected, as these cannot be guaranteed invariant
            for v in list(detected):
                detected.update(MUCT(self.G, v))
                print(f"After adding MUCT nodes, detected set is {detected}"  )
            # Reset the arms that are not guaranteed to be invariant to this change
            for a in set(self.arms) - set(self._structural_resets(detected)):
                arm_index = self.arm_to_index[a]
                self.reset_arm(arm_index)
            
            # Reset all CPD's  of detected nodes
            for d in detected:
                self.cpds[d] = {}
        
    def _structural_resets(self, detected: set[str]) -> None:
        print("Detected", detected)
        invariant_arms = []
        seen_intervention_sets = {}
        S =[f"S_{i}" for i in range(len(detected))]
        S_edges = [(s, d) for s, d in zip(S, detected)]

        for  arm in self.arms:
            intervention_set = frozenset(var for var, _ in arm)
            if intervention_set not in seen_intervention_sets:
                D = CausalDiagram(
                    nodes=self.G.nodes | set(S),
                    directed_edges=self.G.directed_edges.copy() + S_edges,
                    bidirected_edges=self.G.bidirected_edges.copy(),
                )
                D_i = D.do(intervention_set=intervention_set)
                # Check if Y d-separated from S under this intervention
                seen_intervention_sets[intervention_set] = D_i.d_separated({self.reward_node}, set(S), set())

            if seen_intervention_sets[intervention_set]: 
                invariant_arms.append(arm)
            
        return invariant_arms
        
                
    def reset_arm(self, idx: int) -> None:
        self.resat_arms[self.arms[idx]].append(self.t)
        print(f"Resetting arm {self.arms[idx]} due to detected shift")
        self.means[idx] = 0.0
        self.arm_samples[idx] = 0
        
    @override
    def reset(self):
        super().reset()
        self.cpds = {node: {} for node in self.nodes} 
        self.resat_arms = {arm: [] for arm in self.arms}
        self.detected_nodes = {node : [] for node in self.nodes}