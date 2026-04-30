from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
from cmab.algorithms.cpd.rbocpd import RBOCPD
import numpy as np

class RBOCPDUCBAgent(PomisUCBAgent):
    def __init__(
        self,
        reward_node: str,
        G: CausalDiagram,
        arms: list[Intervention],
        c: float,  # UCB exploration parameter
        atomic: bool,
        gamma: float,  # RBOCPD switching rate (optional, default is 1/horizon)
        reset_all: bool = True,  # whether to reset all arms or just the one that triggered the change point
    ):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.means = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.reset_all = reset_all

        self.gamma = gamma

        self.cpds = [RBOCPD(gamma=gamma) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis

    @override
    def _update(self, arm: Intervention, observation: Observation) -> None:
        super()._update(arm, observation)

        arm_index = self.arm_to_index[arm]
        reward = observation[self.reward_node]

        self.cpds[arm_index].update(reward)
        drift_detected = self.cpds[arm_index].drift_detected

        if drift_detected:
            print(f"Step {self.t}: Change point detected for arm {arm}!")
            if self.reset_all: # Reset all arms
                self.means = np.zeros(self.n_arms, dtype=float)
                self.arm_samples = np.zeros(self.n_arms, dtype=int)
                for cpd in self.cpds:
                    cpd.reset()
                for a in self.arms:
                    self.resat_arms[a].append(self.t)
            
            else: # Reset only the arm that triggered the alarm
                self.means[arm_index] = 0.0
                self.arm_samples[arm_index] =0
                self.cpds[arm_index].reset()
                self.resat_arms[arm].append(self.t)


    def reset(self) -> None:
        super().reset()
        for cpd in self.cpds:
            cpd.reset()
        self.resat_arms = {arm : [] for arm in self.arms}