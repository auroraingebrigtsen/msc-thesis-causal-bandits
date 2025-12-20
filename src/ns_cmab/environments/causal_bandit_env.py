from typing import Dict, Callable, List
import numpy as np
from collections import deque
from abc import ABC, abstractmethod

class Mechanism(ABC):
    def __init__(self, func: Callable, exogenous_dist,  seed):
        self.func = func
        self.exogenous_dist = exogenous_dist
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def get_expected_exogenous(self):
        return self.exogenous_dist.mean()
        

    def exogenous_shift(self):
        self.rng = np.random.default_rng(seed=self.seed + 1)

    def __call__(self, parents:dict[str, float]):
        exogenous = self.rng.uniform(0,1)
        return self.func(parents, exogenous)
    

class SCM:
    """
    graph: a parent map (dictionary with node -> parents)
    """
    def __init__(self, graph: Dict[str, List[str]], mechanisms:Dict[str, Mechanism]):
        self.graph = graph
        self.mechanisms = mechanisms

        self.V = len(graph)
        self.topological_order = self._topological_sort()

    # Function to perform Kahn's Algorithm (somewhat stolen from geeksfromgeeks, may implement myself later??)
    def _topological_sort(self):
        in_degree = {node: len(parents) for node, parents in self.graph.items()}

        children = {node: [] for node in self.graph}
        for node, parents in self.graph.items():
            for parent in parents:
                children[parent].append(node)

        # Queue for vertices with 0 in-degree
        queue = deque([i for i in list(self.graph.keys()) if in_degree[i] == 0])
        topo_order = []

        while queue:
            u = queue.popleft()
            topo_order.append(u)

            # Decrease in-degree for adjacent vertices
            for v in children[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return topo_order

    def sample(self, interventions=None):
        "Interventions is a dict of node -> value"
        if interventions is None:
            interventions = {}

        values = {}
        for node in self.topological_order:
            if node in interventions.keys():
                value = interventions[node]
            else:
                value = self.mechanisms[node](values)
            values[node] = value

        return values
    


class CausalBanditEnv:
    def __init__(self, scm, reward_node: str, seed=42):
        self.scm = scm
        self.reward_node = reward_node
        self._step = 0
        self.prob_distribution_shift = 0.01  # Probability of distribution shift at each step
        self.rng = np.random.default_rng(seed=seed)
        self.state = None
        self.action_space = {node: [1,2,3] for node in scm.graph.keys() if node != reward_node}

    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {"steps": self._step}
    
    def _get_optimal_action(self):
        expected_exogenous = {}
        for node, value in self.action_space.items():
            reward = self.scm.sample(interventions=action)[self.reward_node]

    def step(self, action: dict[str, float]):
        self._step += 1
        reward = self.scm.sample(interventions=action)[self.reward_node]
        return self._get_obs(), reward, False, False, self._get_info()  # observation, reward, terminated, truncated, info
        
    def _distribution_shift(self):
        if self.rng.uniform(0, 1) < self.prob_distribution_shift:
            node_to_shift = self.rng.choice(list(self.scm.graph.keys()))  # Decide which arm to shift
            self.scm.mechanisms[node_to_shift].exogenous_shift()

    def reset(self):
        self.step = 0