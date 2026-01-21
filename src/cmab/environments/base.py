from cmab.scm.scm import SCM
import numpy as np
from cmab.typing import InterventionSet
import itertools
from abc import ABC, abstractmethod

class BaseCausalBanditEnv(ABC):
    def __init__(self, scm: SCM, reward_node: str, seed:int=42, atomic: bool = False, non_intervenable: list[str] = []):
        self.scm = scm
        self.reward_node = reward_node
        self._step = 0
        self.rng = np.random.default_rng(seed=seed)
        self.state = None
        self.action_space: list[InterventionSet] = self._init_action_space(atomic=atomic, non_intervenable=non_intervenable)

    @staticmethod
    def _action_sort_key(action: InterventionSet):
        return (len(action), tuple(sorted(action)))

    def _init_action_space(self, atomic: bool, non_intervenable: list[str]) -> list[InterventionSet]:
        """Adds all combinations of possible interventions except for the reward node.
        If atomic is True, only single node interventions are added. 
        The list of non_intervenable nodes are excluded from the action space.
        """
        action_space = set()
        variables = [v for v in self.scm.V if v != self.reward_node]

        if len(non_intervenable) > 0:
            variables = [v for v in variables if v not in non_intervenable]

        if atomic:  # add to the action set: {(var, val)} for each var and val
            for var in variables:
                for val in self.scm.domains[var].support():
                    action_space.add(frozenset({(var, val)}))

        else:  # add to the action set: {(var,1 val1), (var1, val2), ... }
            for k in range(1, len(variables) + 1):
                for subset in itertools.combinations(variables, k):
                    assignments = [()]
                    for var in subset:
                        new_assignments = []

                        for partial in assignments:
                            for val in self.scm.domains[var].support():
                                new_assignments.append(partial + ((var, val),))

                        assignments = new_assignments

                    for assignment in assignments:
                        action_space.add(frozenset(assignment))

        return sorted(action_space, key=self._action_sort_key)

    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {"steps": self._step}
    
    def get_optimal_action(self):
        expected_rewards = {}
        for action in self.action_space:
            expected_reward = self.scm.sample(intervention_set=action, use_mean=True)[self.reward_node]
            expected_rewards[action] = expected_reward
        
        return max(expected_rewards, key=expected_rewards.get)  # Return the key with the highest expected reward

    def get_optimal_expected_reward(self):
        optimal_action = self.get_optimal_action()
        return self.scm.sample(intervention_set=optimal_action, use_mean=True)[self.reward_node]

    @abstractmethod
    def step(self, action: InterventionSet):
        pass

    @abstractmethod
    def reset(self):
        pass