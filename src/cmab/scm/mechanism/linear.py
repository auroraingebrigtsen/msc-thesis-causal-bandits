from .base import BaseMechanism
from typing import Any

class LinearMechanism(BaseMechanism):
    def __init__(self, v_parents: list[str], u_parents: list[str], weights: dict[str, float]):
        """Linear mechanism: Y = w1*X1 + w2*X2 + ... + U1 + U2 + ...
        where U is the exogenous noise term(s) (assumed to be additive)
        Args:
            v_parents (list[str]): List of parent variable names
            u_parents (list[str]): List of exogenous parent variable names
            weights (dict[str, float]): Weights for each parent variable
        """
        super().__init__(v_parents=v_parents, u_parents=u_parents)
        self.initial_weights = weights
        self.weights = weights.copy()

    def __call__(self, v_vals: dict[str, Any], u_vals: dict[str, Any]) -> Any:
        parent_sum = sum(self.weights[parent] * v_vals[parent] for parent in self.v_parents)
        u_sum = sum(self.weights[u_parent] * u_vals[u_parent] for u_parent in self.u_parents)
        return parent_sum + u_sum
    
    def reset(self) -> None:
        self.weights = self.initial_weights.copy()