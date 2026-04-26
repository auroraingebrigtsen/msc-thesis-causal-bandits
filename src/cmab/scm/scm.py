from typing import Mapping

from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.typing import MechanismChangeEvent, ShiftEvent, LinearMechanismChangeEvent
from cmab.scm.distribution.uniform import Uniform
from .distribution.base import BaseDistribution
from .mechanism.base import BaseMechanism as Mechanism
from .causal_diagram import CausalDiagram
from cmab.typing import  Intervention
import numpy as np
from cmab.utils.graphs.topological_sort import topological_sort
from itertools import product

class SCM:
    def __init__(self, 
                 U: list[str], 
                 V: list[str], 
                 P_u_marginals: Mapping[str, BaseDistribution], 
                 F: Mapping[str, Mechanism],
                seed: int = 42,
                 ):
        self.U = U   # List of exogenous variables
        self.V = V  # List of endogenous variables
        self.P_u_marginals = P_u_marginals  # Marginal distributions for exogenous variables
        self.F = F  # Mechanisms for each endogenous variable
        self.V_topological_order = topological_sort(V, [(parent, v) for v in V for parent in F[v].v_parents])  # Topological order of endogenous variables
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, intervention: Intervention = set(), u_values: dict[str, float] = None) -> dict[str, float]:

        # Sample exogenous variables
        if u_values is None:
            u_values = {u:  self.P_u_marginals[u].sample(self.rng) for u in self.U}

        # Go over endogenous variables in topological order and compute their values
        values = {}
        for node in self.V_topological_order:  
            if any(intervention[0] == node for intervention in intervention):
                value = next(intervention[1] for intervention in intervention if intervention[0] == node)
            else:
                v_parents = self.F[node].v_parents
                u_parents = self.F[node].u_parents

                # Collect parent values and exogenous values
                v_vals = {parent: values[parent] for parent in v_parents}
                u_vals = {u_parent: u_values[u_parent] for u_parent in u_parents}

                value = self.F[node](v_vals, u_vals)

            values[node] = value

        return values
    
    def support(self, variable:str) -> set[int]:
        """TODO: only works for discrete variables now
        Compute the support of a variable by enumerating all combinations of exogenous variables and applying the structural equations.
        """
        support = set()

        u_values = {u: self.P_u_marginals[u].support() for u in self.U}
        for u_combination in product(*u_values.values()):
            u_assignment = dict(zip(self.U, u_combination))
            v_values = self.sample(u_values=u_assignment)
            support.add(v_values[variable])

        return support

    def expected_value(self, variable:str, intervention: Intervention = set()) -> float:
        """Compute the expected values of a variable Y given an intervention set, that is, E[Y | do(X=x)], when all exogenous variables are in a given interval."""
        def p_u(u_values: dict[str, float]) -> float:
            """Compute the probability of a given assignment to the exogenous variables Ex: {'U_X_1': 0, 'U_X_2': 1, 'U_Z_1': 0, 'U_Z_2': 1, 'U_Y': 0}"""
            p = 1.0
            for u in self.U:
                val = u_values[u]
                p *= float(self.P_u_marginals[u].prob(val))
            return p
        
        expected = 0.0

        u_values = {u: self.P_u_marginals[u].support() for u in self.U}
        for u_combination in product(*u_values.values()):
            u_assignment = dict(zip(self.U, u_combination))
            prob = p_u(u_assignment)

            if prob == 0.0:
                continue

            v_values = self.sample(intervention=intervention, u_values=u_assignment)
            y = v_values[variable]

            expected += prob * float(y)
        return float(expected)

    def apply_change_event(self, event: ShiftEvent | MechanismChangeEvent | LinearMechanismChangeEvent):
        """Apply a change event to the SCM by updating the relevant distribution or mechanism."""
        if isinstance(event, ShiftEvent):
            self.apply_shift(event)
        elif isinstance(event, MechanismChangeEvent):
            self.apply_mechanism_change(event)
        elif isinstance(event, LinearMechanismChangeEvent):
            self.apply_linear_mechanism_change(event)

    def apply_shift(self, event: ShiftEvent):
        """Apply a shift event to the SCM by updating the relevant exogenous distribution."""
        dist = self.P_u_marginals[event.variable]
        dist.update_parameters(event.new_param)

    def apply_mechanism_change(self, event: MechanismChangeEvent):
        """Apply a mechanism change event to the SCM by updating the relevant mechanism."""
        new_mechanism = CustomMechanism(
            v_parents=self.F[event.variable].v_parents, 
            u_parents=self.F[event.variable].u_parents,
            f=lambda v, u: eval(event.new_mechanism)
        )
        self.F[event.variable] = new_mechanism

    def apply_linear_mechanism_change(self, event: LinearMechanismChangeEvent):
        """Apply a linear mechanism change event to the SCM by updating the relevant mechanism."""
        old_mechanism = self.F[event.variable]
        new_mechanism = LinearMechanism(
            v_parents=old_mechanism.v_parents, 
            u_parents=old_mechanism.u_parents,
            weights=event.new_weights
        )
        self.F[event.variable] = new_mechanism

    def get_causal_diagram(self) -> CausalDiagram:
        """Get the causal diagram (with bidirected edges for UCs) and no exogenous nodes."""
        directed_edges = []
        bidirected_edges = []
        for v in self.V:
            for parent in self.F[v].v_parents:
                directed_edges.append((parent, v))
        for u in self.U:
            children = [v for v in self.V if u in self.F[v].u_parents]
            if len(children) > 1:
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        bidirected_edges.append((children[i], children[j], u))

        return CausalDiagram(nodes=set(self.V), directed_edges=directed_edges, bidirected_edges=bidirected_edges)

    def reset(self, seed:int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        for u in self.U:
            self.P_u_marginals[u].reset()
        for v in self.V:
            self.F[v].reset()