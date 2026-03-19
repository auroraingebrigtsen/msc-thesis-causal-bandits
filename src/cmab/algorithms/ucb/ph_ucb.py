# A  basic Page Hinkley UCB algorithm
from typing import override
from cmab.algorithms.ucb.ucb_base import UCBAgent
import numpy as np
from cmab.typing import Intervention, Observation
from river import drift

class PageHinkleyUCBAgent(UCBAgent):
    def __init__(
        self,
        reward_node: str,
        arms: list[Intervention],
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
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis
        #self.test = ['X', 'X', 'X']

    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        arm_index = self.arm_to_index[arm]
        reward = observation[self.reward_node]

        self.cpds[arm_index].update(reward)

        drift_detected = self.cpds[arm_index].drift_detected

        #if self.t > 1 and self.t < 2000 and self.t % 500 == 0:
        if drift_detected:
            print(f"Step {self.t}: Change point detected for arm {arm}!")
            if self.reset_all: # Reset all arms
                self.estimates = np.zeros(self.n_arms)
                self.arm_samples = np.zeros(self.n_arms, dtype=int)
                self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
                for a in self.arms:
                    self.resat_arms[a].append(self.t)
            
            else: # Reset only the arm that triggered the alarm
                self.estimates[arm_index] = 0.0
                self.arm_samples[arm_index] =0.0 # int(0.5 * self.arm_samples[arm_index])
                self.cpds[arm_index] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
                self.resat_arms[arm].append(self.t)


    def reset(self) -> None:
        super().reset()
        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}