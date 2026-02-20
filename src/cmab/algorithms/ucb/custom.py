from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.algorithms.ucb.ucb_base import UCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import InterventionSet, Observation
from collections import defaultdict
from river import drift
from itertools import product

class MyFirstAtomicAgent(UCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float=2, 
                 delta:float=0.5, lambda_:float=5.0, min_samples_for_detection:int=10):
        super().__init__(reward_node, arms, c)
        self.G = G
        self.nodes = list(G.nodes)
        self.parents = {node: list(G.Pa({node}, include_self=False)) for node in self.nodes}

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        self.cpds = self._init_cpds()
        self.change_points = {node: [] for node in self.nodes}  # Keep track of detected change points for analysis 
        self.test = ['X', 'Z', 'X']
        
    def _init_cpds(self):
        cpds = defaultdict(dict)
        for node in self.nodes:
            parents = self.parents[node]
            parent_cfgs = list(product([0, 1], repeat=len(parents)))  # Currently only binary variables, TODO: extend to use domains
            for cfg in parent_cfgs:
                cpds[node][cfg] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)

        return cpds

    @override
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        super()._update(arm, observation)

        detected = set()
        if self.t > 1 and self.t < 2000 and self.t % 500 == 0:
                print(f"\nStep {self.t}: Change point detected for nodes: {self.test[self.t//500 - 1]}!")
                detected.add(self.test[self.t//500 - 1])

        # for node in self.nodes:
        #     if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
        #         continue

        #     cfg = tuple(observation[parent] for parent in self.parents[node])
        #     self.cpds[node][cfg].update(observation[node])
            
        #     if self.cpds[node][cfg].drift_detected:
        #         print(f"\nStep {self.t}: Change point detected for node {node}!")
        #         detected.add(node)
        #         # Reset CPD state for this node
        #         self.cpds[node][cfg] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) 
        #         # Consider adding some of the previous observations to the new CPD state to make it more robust, but for now we just reset it.

        for node in self.nodes:
            self.change_points[node].append(int(node in detected))

        if len(detected) > 0:
            self._update_on_change_point(detected)


    def _update_on_change_point(self, detected: set[str]) -> None:

        # We know which variable was alarmed, it must be the exogenous variabe of this one that shifted. 
        shifted = set()
        print(f"Detected change points for nodes: {detected}" f" at step {self.t}")
        # Find exogenous variables of detected nodes
        for node in detected:
            exogenous = {u for (u, v) in self.G.noise_vars if v == node}
            shifted.update(exogenous)
        # TODO: In this first situation we assume markovianity. Extend, but need to think more how this will work
        print(f"Shifted exogenous variables: {shifted}" )
        # Find affected variables: those that are not d-separated from the shifted exogenous variables after intervening on themselves.
        for arm_index, arm in enumerate(self.arms):
            intervention_set = {var for var, _ in arm}
            sub = self.G.do(intervention_set=intervention_set)

            # If reward is not d-separated from any shifted exogenous var, reset this arm
            reset_arm = any(
                not sub.d_separated({self.reward_node}, {u}, set())
                for u in shifted
            )

            if reset_arm:
                print(f"Resetting arm {arm} (index {arm_index}) due to detected shift")
                self.estimates[arm_index] = 0.5
                self.arm_samples[arm_index] = 0
        
        print(f"Actions: {self.arms}")
        print(f"Updated estimates: {self.estimates}")
        print(f"Updated arm samples: {self.arm_samples}")        
        
    @override
    def reset(self):
        super().reset()
        self.cpds = self._init_cpds()
        self.change_points = {node: [] for node in self.nodes}


# class MyFirstAgent(PomisUCBAgent):
#     def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float, delta:float, lambda_:float, min_samples_for_detection:int):
#         super().__init__(reward_node, G, arms, c)
#         self.nodes = list(G.nodes - {reward_node})

#         self.cpds = {node: PageHinkleyCPD(node, G.Pa({node}, include_self=False), 
#                                           delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection) 
#                                           for node in self.nodes}

#     @override
#     def _update(self, arm: InterventionSet, observation: Observation) -> None:
#         super()._update(arm, observation)
#         detected = set()
#         for node in self.nodes:
#             if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
#                 continue
#             cpd = self.cpds[node]
#             change_point = cpd.update(observation)

#             if change_point:
#                 print(f"Step {self.t}: Change point detected for node {node}!")
#                 detected.add(node)
        
#         if detected:
#             self._update_on_change_point(detected)

#     def _update_on_change_point(self, detected: set[str]) -> None:

#         # We know which variable was alarmed, it must be the exogenous variabe
#         # of this one that shifted. 
#         shifted = set()
#         for node in detected:
#             exogenous = {u for (u, v) in self.G.noise_vars if v == node}
#             shifted.update(exogenous)
#         # TODO: In this first situation we assume markovianity. Extend, but need to think more how this will work

#         affected_vars = set()
#         for node in self.nodes:
#             sub = self.G.do(intervention_set={node})
#             for shifted_u in shifted:
#                 if not sub.d_separated({node}, {shifted_u}, set()):
#                     affected_vars.add(node)
        
#         for node in affected_vars:
#             arm_index = self.arm_to_index.get(node)  # TODO fix so it updates all sets involving node
#             self.estimates[arm_index] = 0.0
#             self.arm_samples[arm_index] = 0
#             self.cpds[node].reset()
