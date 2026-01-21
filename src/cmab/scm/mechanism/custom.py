from .base import BaseMechanism
from typing import Any, Callable

class CustomMechanism(BaseMechanism):
    """General mechanism defined by a function f, for more specific mechanisms where other generic implementations are not suitable. 
    Args:
        v_parents (list[str]): List of parent variable names
        u_parents (list[str]): List of exogenous parent variable names
        f (Callable[[dict[str, Any], dict[str, Any]], Any]): Lambda function defining the mechanism
    """
    def __init__(self, v_parents: list[str], u_parents: list[str], f: Callable[[dict[str, Any], dict[str, Any]], Any]):
        super().__init__(v_parents=v_parents, u_parents=u_parents)
        self.f = f
        self.ex

    def __call__(self, v_vals: dict[str, Any], u_vals: dict[str, Any]) -> Any:
        print(f"CustomMechanism called with v_vals: {v_vals}, u_vals: {u_vals}")
        print(f"Types - v_vals: { {k: type(v) for k, v in v_vals.items()} }, u_vals: { {k: type(v) for k, v in u_vals.items()} }")
        return self.f(v_vals, u_vals)


    def expected(self, v_vals_expected: dict[str, float], u_vals_expected: dict[str, float]) -> float:
        "pass??