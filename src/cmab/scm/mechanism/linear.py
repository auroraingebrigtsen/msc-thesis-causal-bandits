from .base import BaseMechanism
import numpy as np
from typing import Any

class LinearMechanism(BaseMechanism):
    def __init__(self, parents: list[str], u_parents: list[str], weights: list[float], bias:float=0.0):
        """Linear mechanism: Y = w1*X1 + w2*X2 + ... + b + U1 + U2 + ...
        where U is the exogenous noise term(s) (assumed to be additive)
        Args:
            parents (list[str]): List of parent variable names
            u_parents (list[str]): List of exogenous parent variable names
            weights (list[float]): Weights for each parent variable
            bias (float): Bias term
            seed (int): Random seed for reproducibility
        """
        self.v_parents = parents
        self.u_parents = u_parents
        self.weights = weights
        self.bias = bias

    def __call__(self, v_vals: dict[str, Any], u_vals: dict[str, Any]) -> Any:
        parent_sum = sum(self.weights[i] * v_vals[parent] for i, parent in enumerate(self.v_parents))
        u_sum = sum(u_vals[u_parent] for u_parent in self.u_parents)
        return parent_sum + self.bias + u_sum