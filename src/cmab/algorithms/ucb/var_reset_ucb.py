# A  basic Page Hinkley UCB algorithm
from typing import override
from cmab.algorithms.ucb.ucb_base import UCBAgent
import numpy as np
from cmab.typing import Intervention, Observation
from river import drift

class VarResetUCBAgent(UCBAgent):
    def __init__(
        self,
        reward_node: str,
        arms: list[Intervention],
        c: float,  # UCB exploration parameter
        delta: float,  # a small positive value (tolerance) to prevent overreacting to small fluctuations
        lambda_: float,  # the threshold for change detection
        min_samples_for_detection,
    ):
        super().__init__(reward_node, arms, c)
        self.n_arms = len(arms)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection

        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis
        #self.test = ['X', 'X', 'X']

        self.index_to_arm = {i: arm for i, arm in enumerate(arms)}

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
            # Find all arms  intervening on the same variable as arm, and reset those as well, as they likely changed to
            intervention_set = set(var for var, _ in arm)
            for a in self.arms:
                a_intervention_set = set(var for var, _ in a)
                if a_intervention_set == intervention_set: # if they intervene on the same variable(s)
                    print(f"Resetting arm {a} as well due to shared intervention on variables {intervention_set}")
                    self.estimates[arm_index] = 0.0
                    self.arm_samples[arm_index] =0.0 # int(0.5 * self.arm_samples[arm_index])
                    self.cpds[arm_index] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
                    self.resat_arms[arm].append(self.t)


    def reset(self) -> None:
        super().reset()
        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}