# A  basic Page Hinkley UCB algorithm
from cmab.algorithms.ucb.ucb_base import UCBAgent
import numpy as np
from cmab.typing import InterventionSet, Observation
from cmab.algorithms.cpd.ph_cpd import PageHinkleyCPD

class PageHinkleyUCBAgent(UCBAgent):
    def __init__(
        self,
        reward_node: str,
        arms: list[InterventionSet],
        c: float = 2,
        delta: float = 0.5,
        lambda_: float = 5.0,
        min_samples_for_detection: int = 10,
    ):
        super().__init__(reward_node, arms, c)
        self.n_arms = len(arms)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.t = 0

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        # Avoid O(K) list.index lookups
        self._arm_to_index = {arm: i for i, arm in enumerate(self.arms)}

        self.cpds = self._initialize_cpds()

    def _initialize_cpds(self) -> dict[int, PageHinkleyCPD]:
        cpds: dict[int, PageHinkleyCPD] = {}
        for i in range(self.n_arms):
            cpds[i] = PageHinkleyCPD(
                node=self.reward_node,   # <-- critical change
                parents=[],
                delta=self.delta,
                lambda_=self.lambda_,
                min_samples_for_detection=self.min_samples_for_detection,
            )
        return cpds

    def select_arm(self) -> InterventionSet:
        # Ensure each arm tried once
        for i in range(self.n_arms):
            if self.arm_samples[i] == 0:
                return self.arms[i]

        t_for_ucb = max(1, self.t)  # avoid log(0)
        ucb_values = np.empty(self.n_arms, dtype=float)
        for i in range(self.n_arms):
            n_i = self.arm_samples[i]
            bound = np.sqrt(np.log(t_for_ucb) / n_i)
            ucb_values[i] = self.estimates[i] + self.c * bound

        return self.arms[int(np.argmax(ucb_values))]

    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        reward = float(observation[self.reward_node])
        self.t += 1

        arm_index = self._arm_to_index[arm]
        self.arm_samples[arm_index] += 1
        n = self.arm_samples[arm_index]

        # Incremental mean
        self.estimates[arm_index] += (reward - self.estimates[arm_index]) / n

        # Only reward stream for *this* arm goes into *this* CPD instance
        if self.cpds[arm_index].update(observation):
            print(f"Change point detected for arm {arm_index}!")
            self.reset()

    def reset(self) -> None:
        self.t = 0
        self.estimates[:] = 0.0
        self.arm_samples[:] = 0
        for cpd in self.cpds.values():
            cpd.reset()