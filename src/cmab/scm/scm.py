from collections import deque
from typing import Dict, Any, FrozenSet, Mapping
from .pmf.base import BasePmf as Pmf
from .mechanism.base import BaseMechanism as Mechanism
from .domain.base import FiniteDiscreteDomain
from cmab.typing import  InterventionSet
import numpy as np


class SCM:
    def __init__(self, 
                 U: FrozenSet[str], 
                 V: FrozenSet[str], 
                 domains: Mapping[str, FiniteDiscreteDomain], 
                 P_u_marginals: Mapping[str, Pmf], 
                 F: Mapping[str, Mechanism],
                seed: int = 42
                 ):
        self.U = U   # Set of exogenous variables
        self.V = V  # Set of endogenous variables
        self.domains = domains  # Domain for each variable
        self.P_u_marginals = P_u_marginals  # Marginal distributions for exogenous variables
        self.F = F  # Mechanisms for each endogenous variable
        self.V_topological_order = self._topological_sort()  # Topological order of endogenous variables
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    # Function to perform Kahn's Algorithm (somewhat stolen from geeksfromgeeks, may implement myself later??)
    def _topological_sort(self):
        parents = {node: self.F[node].v_parents for node in self.V}
        in_degree = {node: len(parents) for node, parents in parents.items()}

        children = {node: [] for node in self.V}
        for node, parents in parents.items():
            for parent in parents:
                children[parent].append(node)

        # Queue for vertices with 0 in-degree
        queue = deque([i for i in list(self.V) if in_degree[i] == 0])
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

    def sample(self, intervention_set: InterventionSet = set(), use_mean:bool=False) -> Dict[str, Any]:
        values = {}
        for node in self.V_topological_order:
            if any(intervention[0] == node for intervention in intervention_set):
                value = next(intervention[1] for intervention in intervention_set if intervention[0] == node)
            else:
                v_parents = self.F[node].v_parents
                u_parents = self.F[node].u_parents

                # Collect parent values and sample exogenous values
                v_vals = {parent: values[parent] for parent in v_parents}
                if use_mean:
                    u_vals = {u_parent: self.P_u_marginals[u_parent].mean() for u_parent in u_parents}
                else:
                    u_vals = {u_parent: self.P_u_marginals[u_parent].sample(self.rng) for u_parent in u_parents}

                value = self.F[node](v_vals, u_vals)

            values[node] = value

        return values

    def exogenous_distribution_shift(self, max_delta: float = 0.2) -> None:
        u_to_shift = self.rng.choice(list(self.U))
        print(f"Shifting distribution of latent variable {u_to_shift}. Values before shift: {self.P_u_marginals[u_to_shift].p}")
        self.P_u_marginals[u_to_shift].distribution_shift(rng=self.rng, max_delta=max_delta)
        print(f"Values after shift: {self.P_u_marginals[u_to_shift].p}")

    def reset(self):
        self.seed += 1
        self.rng = np.random.default_rng(seed=self.seed)