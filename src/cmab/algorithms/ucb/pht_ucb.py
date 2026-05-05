from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from typing import override
from cmab.scm.causal_diagram import CausalDiagram
from cmab.typing import Intervention, Observation
from river import drift
import numpy as np

class PageHinkleyUCBAgent(PomisUCBAgent):
    def __init__(
        self,
        reward_node: str,
        G: CausalDiagram,
        arms: list[Intervention],
        c: float,  # UCB exploration parameter
        atomic: bool,
        delta: float,  # a small positive value (tolerance) to prevent overreacting to small fluctuations
        lambda_: float,  # the threshold for change detection
        min_samples_for_detection,
        reset_all: bool = True , # whether to reset all arms or just the one that triggered the change point
        alpha:float=0.6931, # probability of exploring randomly vs using UCB 
        seed: int = 42, # random seed for any randomization in the algorithm 
    ):
        super().__init__(reward_node=reward_node, G=G, arms=arms, c=c, atomic=atomic)
        self.means = np.zeros(self.n_arms, dtype=float)
        self.arm_samples = np.zeros(self.n_arms, dtype=int)
        self.reset_all = reset_all

        self.delta = delta
        self.lambda_ = lambda_
        self.min_samples_for_detection = min_samples_for_detection
        self.alpha = alpha
        self.seed = seed

        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}  # Keep track of detected change points for analysis
        self.rng = np.random.default_rng(seed)
    
    @override
    def select_arm(self) -> int:
        for i in range(self.n_arms):   # ensure each arm is tried once
            if self.arm_samples[i] == 0:
                return self.arms[i]

        if self.rng.uniform() < self.alpha: 
            return self.rng.choice(self.arms)
        
        ucb_values = []
        for arm in range(self.n_arms): 
            n_arm = self.arm_samples[arm]
            t = np.sum(self.arm_samples)  
            bound = np.sqrt(np.log(t)/n_arm)
            ucb_values.append(self.means[arm] + self.c*bound)
        return self.arms[np.argmax(ucb_values)]

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
                self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
                for a in self.arms:
                    self.resat_arms[a].append(self.t)
            
            else: # Reset only the arm that triggered the alarm
                self.means[arm_index] = 0.0
                self.arm_samples[arm_index] =0.0
                self.cpds[arm_index] = drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection)
                self.resat_arms[arm].append(self.t)

    def reset(self) -> None:
        super().reset()
        self.cpds = [drift.PageHinkley(delta=self.delta, threshold=self.lambda_, min_instances=self.min_samples_for_detection) for _ in range(self.n_arms)]
        self.resat_arms = {arm : [] for arm in self.arms}
        self.rng = np.random.default_rng(self.seed+1)