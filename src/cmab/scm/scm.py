from collections import deque
from typing import Dict, Any, FrozenSet, Mapping
from .pmf.base import BasePmf as Pmf
from .mechanism.base import BaseMechanism as Mechanism
from .domain.base import FiniteDiscreteDomain
from .causal_diagram import CausalDiagram
from cmab.typing import  InterventionSet
import numpy as np
from cmab.utils.graphs.topological_sort import topological_sort

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
        self.V_topological_order = topological_sort(V, [(parent, v) for v in V for parent in F[v].v_parents])  # Topological order of endogenous variables
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, intervention_set: InterventionSet = set(), expected: bool = False) -> Dict[str, Any]:

        # Sample exogenous variables
        u_values = {u: self.P_u_marginals[u].mean(self.rng) if expected else self.P_u_marginals[u].sample(self.rng) for u in self.U}

        # Go over endogenous variables in topological order and compute their values
        values = {}
        for node in self.V_topological_order:
            if any(intervention[0] == node for intervention in intervention_set):
                value = next(intervention[1] for intervention in intervention_set if intervention[0] == node)
            else:
                v_parents = self.F[node].v_parents
                u_parents = self.F[node].u_parents

                # Collect parent values and exogenous values
                v_vals = {parent: values[parent] for parent in v_parents}
                u_vals = {u_parent: u_values[u_parent] for u_parent in u_parents}

                if expected:
                        value = self.F[node].expected(v_vals, u_vals)
                else:
                    value = self.F[node](v_vals, u_vals)

            values[node] = value

        return values

    def exogenous_distribution_shift(self, max_delta: float = 0.2) -> None:
        u_to_shift = self.rng.choice(list(self.U))
        print(f"Shifting distribution of latent variable {u_to_shift}. Values before shift: {self.P_u_marginals[u_to_shift].p}")
        self.P_u_marginals[u_to_shift].distribution_shift(rng=self.rng, max_delta=max_delta)
        print(f"Values after shift: {self.P_u_marginals[u_to_shift].p}")


    def get_causal_diagram(self) -> CausalDiagram:
        directed_edges = []
        bidirected_edges = []
        for v in self.V:
            for parent in self.F[v].v_parents:
                directed_edges.append((parent, v))
        for u in self.U:
            children = [v for v in self.V if u in self.F[v].u_parents]
            if len(children) >= 2:
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        bidirected_edges.append((children[i], children[j], u))
        return CausalDiagram(nodes=self.V, directed_edges=directed_edges, bidirected_edges=bidirected_edges)

    def reset(self):
        self.seed += 1
        self.rng = np.random.default_rng(seed=self.seed)