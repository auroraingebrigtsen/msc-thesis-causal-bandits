from base import Base
import numpy as np

class UCBAgent(Base):
    """
    Args:
    c: float, degree of exploration
    """
    def __init__(self, bandits, c:float=2, random_state: int=42):
        super().__init__(bandits, random_state)
        self.c = c
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)

    def _update(self, action, reward):
        self.action_samples[action] += 1
        num_samples = self.action_samples[action]
        prev_reward = self.estimates[action]
        self.estimates[action] = prev_reward + 1/(num_samples)*(reward - prev_reward)
    
    def select_action(self):
        ucb_estimates = []
        total_runs = np.sum(self.action_samples) + 1 # starting with first run
        for action in range(self.k): 
            action_samples = self.action_samples[action] if self.action_samples[action] != 0 else 1
            bound = np.sqrt(np.log(total_runs)/action_samples)
            ucb_estimates.append(self.estimates[action] + self.c*bound)
        return np.argmax(ucb_estimates)
    
    def reset(self):
        self.estimates = np.zeros(self.k)
        self.action_samples = np.zeros(self.k)
        self.regret = []
        self.averaged_regret = []
        self.random_state += 1
        np.random.seed(self.random_state)
