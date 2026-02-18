# A  basic Page Hinkley UCB algorithm
from cmab.algorithms.ucb.ucb_base import UCBAgent
import numpy as np
from cmab.typing import InterventionSet, Observation
from river import drift

class PageHinkleyUCBAgent(UCBAgent):
    def __init__(
        self,
        reward_node: str,
        arms: list[InterventionSet],
        c: float,  # UCB exploration parameter
        delta: float,  # a small positive value (tolerance) to prevent overreacting to small fluctuations
        lambda_: float,  # the threshold for change detection
        min_samples_for_detection,
        reset_all: bool = True  # whether to reset all arms or just the one that triggered the change point
    ):
        super().__init__(reward_node, arms, c)
        self.n_arms = len(arms)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection
        self.reset_all = reset_all

        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]

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
        super()._update(arm, observation)

        arm_index = self.arms.index(arm)
        reward = observation[self.reward_node]

        self.cpds[arm_index].update(reward)

        if self.cpds[arm_index].drift_detected:
            print(f"Step {self.t}: Change point detected for arm {arm}!")
            if self.reset_all:
                # Reset estimates and samples 
                self.estimates = np.zeros(self.n_arms)
                self.arm_samples = np.zeros(self.n_arms)
                # Reset CPDs
                self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
            
            else:
                # Reset only the affected arm
                self.estimates[arm_index] = 0.0
                self.arm_samples[arm_index] = 0
                self.cpds[arm_index] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
    

    def reset(self) -> None:
        super().reset()
        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        