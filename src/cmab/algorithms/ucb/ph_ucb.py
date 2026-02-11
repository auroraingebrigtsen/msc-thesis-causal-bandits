# A  basic Page Hinkley UCB algorithm
from cmab.algorithms.ucb.ucb_base import UCBAgent
import numpy as np
from cmab.typing import InterventionSet, Observation
from cmab.algorithms.cpd.ph_cpd import PageHinkleyCPD, ArmLevelPageHinkleyCPD
from collections import defaultdict

class PageHinkleyUCBAgent(UCBAgent):
    def __init__(
        self,
        reward_node: str,
        arms: list[InterventionSet],
        c: float,  # UCB exploration parameter
        delta: float,  # a small positive value (tolerance) to prevent overreacting to small fluctuations
        lambda_: float,  # the threshold for change detection
        min_samples_for_detection,
    ):
        super().__init__(reward_node, arms, c)
        self.n_arms = len(arms)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.t = 0
        self.total_steps = 0

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        self.cpds = defaultdict(
            lambda: ArmLevelPageHinkleyCPD(
                delta=self.delta,
                lambda_=self.lambda_,
                min_samples_for_detection=self.min_samples_for_detection
            )
        )

    def select_arm(self) -> InterventionSet:
        # Ensure each arm tried once
        for i in range(self.n_arms):
            if self.arm_samples[i] == 0:
                return self.arms[i]

        ucb_values = np.empty(self.n_arms, dtype=float)
        for i in range(self.n_arms):
            n_i = self.arm_samples[i]
            bound = np.sqrt(np.log(self.t) / n_i)
            ucb_values[i] = self.estimates[i] + self.c * bound

        return self.arms[int(np.argmax(ucb_values))]

    def _update(self, arm: InterventionSet, observation: Observation) -> None:
        reward = float(observation[self.reward_node])
        self.t += 1
        self.total_steps += 1

        arm_index = self.arms.index(arm)
        self.arm_samples[arm_index] += 1
        n = self.arm_samples[arm_index]

        self.estimates[arm_index] += (reward - self.estimates[arm_index]) / n

        if self.cpds[arm_index].update(reward):
            print(f"Step {self.total_steps}: Change point detected for arm {arm}!")
            self.reset_arms()
        
    def reset_arms(self) -> None:
        """Resets the estimates and samples for all arms."""
        self.t = 0
        self.estimates[:] = 0.0
        self.arm_samples[:] = 0
        self.cpds.clear()

    def reset(self) -> None:
        self.reset_arms()
        self.total_steps = 0