from .base import BaseMechanism

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

    def reset(self) -> None:
        pass  # No parameters to reset in the XOR mechanism