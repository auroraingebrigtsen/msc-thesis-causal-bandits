from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
from cmab.algorithms.cpd.rbocpd import RBOCPD
import numpy as np

class RBOCPDSrUCBAgent(PomisUCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[Intervention], gamma:float, c:float=np.sqrt(2), atomic:bool=False):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.G = G
        self.nodes = list(G.nodes)
        self.parents = {node: list(G.Pa({node}, include_self=False)) for node in self.nodes}
        
        self.gamma = gamma

        self.cpds = {node: {} for node in self.nodes} # cpds[node][parent_cfg] gives the drift detector for that node and parent configuration
        self.resat_arms = {arm: [] for arm in self.arms}  # Keep track of detected change points for analysis 
        
    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        detected = set()

        for node in self.nodes:
            if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
                continue

            cfg = tuple(observation[parent] for parent in self.parents[node])
            if cfg not in self.cpds[node]:
                self.cpds[node][cfg] = RBOCPD(gamma=self.gamma)
            self.cpds[node][cfg].update(observation[node])
            
            if self.cpds[node][cfg].drift_detected:
                print(f"\nStep {self.t}: Change point detected for node {node}!")
                detected.add(node)
                # Reset all CPD's associated with this node
                for cfg in self.cpds[node]: # reset all parent contexts for this node (if one changes, the other ones do to)
                    self.cpds[node][cfg].reset()
                    # Could consider adding some of the previous observations to the new CPD state to make it more robust, but for now we just reset it.
        self.t += 1

        if len(detected) > 0:
            # Add variables sharing an UC with nodes in detected, to detected, as these cannot be guaranteed invariant
            for v in list(detected):
                detected.update(self.G.bidirected_neighbors[v])
            # Reset the arms that are not guaranteed to be invariant to this change
            for a in set(self.arms) - set(self._structural_resets(detected)):
                arm_index = self.arm_to_index[a]
                self.reset_arm(arm_index)


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
                # Check if Y d_separated from S
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