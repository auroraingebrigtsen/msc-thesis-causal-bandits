from .base import BaseMechanism
from typing import Any

class XORMechanism(BaseMechanism):
    def __init__(self, v_parents: list[str], u_parents: list[str]):
        """XOR mechanism: Y = X1 XOR X2 XOR ... XOR  U1 XOR U2 ...
        where U is the exogenous noise term(s) (assumed to be additive)
        Args:
            v_parents (list[str]): List of parent variable names
            u_parents (list[str]): List of exogenous parent variable names
        """
        super().__init__(v_parents=v_parents, u_parents=u_parents)

    def __call__(self, v_vals: dict[str, int], u_vals: dict[str, int]) -> int:
        result = 0
        for parent in self.v_parents:
            result ^= v_vals[parent]
        for u_parent in self.u_parents:
            result ^= u_vals[u_parent]
        return result
    
    def expected(self, v_vals_expected: dict[str, float], u_vals_expected: dict[str, float]) -> float:
        """Compute the expected value of the XOR mechanism given expected values of parents. Values in {0,1}.
        Args:
            v_vals_expected (dict[str, float]): Expected values of parent variables
            u_vals_expected (dict[str, float]): Expected values of exogenous parent variables
        Returns:
            float: Expected value of the XOR mechanism
        """
        
        prob_one = 0.0
        for parent in self.v_parents:
            p = v_vals_expected[parent]
            prob_one = prob_one * (1 - p) + (1 - prob_one) * p
        for u_parent in self.u_parents:
            p = u_vals_expected[u_parent]
            prob_one = prob_one * (1 - p) + (1 - prob_one) * p
        return prob_one