from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.algorithms.ucb.ucb_base import UCBAgent
from typing import override
from cmab.algorithms.cpd.ph_cpd import PageHinkleyCPD
from cmab.scm .causal_diagram import CausalDiagram
from cmab.typing import InterventionSet, Observation
from collections import defaultdict


class MyFirstAgent(PomisUCBAgent):
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float, delta:float, lambda_:float, min_samples_for_detection:int):
        super().__init__(reward_node, G, arms, c)
        self.nodes = list(G.nodes - {reward_node})

        self.cpds = {node: PageHinkleyCPD(node, G.Pa({node}, include_self=False), 
                                          delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection) 
                                          for node in self.nodes}

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
                print(f"Step {self.t}: Change point detected for node {node}!")
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
    def __init__(self, reward_node:str, G: CausalDiagram, arms: list[InterventionSet], c:float=2, 
                 delta:float=0.5, lambda_:float=5.0, min_samples_for_detection:int=10):
        super().__init__(reward_node, arms, c)
        self.G = G
        self.nodes = list(G.nodes)

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        self.cpds = {node: PageHinkleyCPD(node, G.Pa({node}, include_self=False), delta=self.delta, lambda_=self.lambda_, 
                                          min_samples_for_detection=self.min_samples_for_detection) for node in self.nodes}

    @override
    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        super()._update(arm, observation)
        print(f"\n\nStep {self.t}: Selected arm {arm}")
        print(f"Observation: {observation}")


        detected = set()
        for node in self.nodes:
            if any(var == node for var, _ in arm): # Dont update cpd for intervened nodes
                continue
            
            if self.cpds[node].update(observation):
                print(f"Step {self.t}: Change point detected for node {node}!")
                detected.add(node)
                self.cpds[node].reset()  # Reset CPD state for this node
            
        
        if detected:
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
        affected_vars = set()
        for node in self.nodes:
            sub = self.G.do(intervention_set={node})
            for shifted_u in shifted:
                print(f"D-separation test for node {self.reward_node} and shifted exogenous {shifted_u}: {sub.d_separated({self.reward_node}, {shifted_u}, set())}")
                if not sub.d_separated({self.reward_node}, {shifted_u}, set()): # If not d-separated, node is affected
                    affected_vars.add(node)
        
        print(f"Affected variables: {affected_vars}\n\n")
        
        for node in affected_vars:
            # find all arms that involve this node
            for arm in self.arms:
                if any(var == node for var, _ in arm):
                    # reset UCB estimates
                    arm_index = self.arms.index(arm)
                    print(f"Resetting arm {arm} (with index {arm_index}) due to change point in node {node}")
                    self.estimates[arm_index] = 0.0
                    self.arm_samples[arm_index] = 0
            

    @override
    def reset(self):
        super().reset()
        for cpd in self.cpds.values():
            cpd.reset()