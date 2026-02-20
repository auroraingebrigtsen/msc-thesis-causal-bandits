from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.algorithms.ucb import UCBAgent, SlidingWindowUCBAgent
from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np

### The setup from Task 1 in Structural Causal Bandits: Where to Intervene? ### 

def main():
    SEED = 42

    V = frozenset({'X_1', 'X_2', 'Z_1', 'Z_2', 'Y'})
    U = frozenset({'U_X_1', 'U_X_2', 'U_Z_1', 'U_Z_2', 'U_Y'})

    domains = {
        'X_1': BinaryDomain(),
        'X_2': BinaryDomain(),
        'Z_1': BinaryDomain(),
        'Z_2': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X_1 = Bernoulli(p=0.54)
    P_X_2 = Bernoulli(p=0.67)
    P_Z_1 = Bernoulli(p=0.54)
    P_Z_2 = Bernoulli(p=0.44)
    P_Y = Bernoulli(p=0.58)
    
    mechanism_X_1 = XORMechanism(v_parents=['Z_1', 'Z_2'], u_parents=['U_X_1'])
    mechanism_X_2 = CustomMechanism(v_parents=['Z_1', 'Z_2'], u_parents=['U_X_2'], 
                                    f=lambda v, u: 1 ^ v['Z_1'] ^ v['Z_2'] ^ u['U_X_2'])
    mechanism_Z_1 = CustomMechanism(v_parents=[], u_parents=['U_Z_1'], f=lambda _, u: u['U_Z_1'])
    mechanism_Z_2 = CustomMechanism(v_parents=[], u_parents=['U_Z_2'], f=lambda _, u: u['U_Z_2'])
    mechanism_Y = CustomMechanism(v_parents=['X_1', 'X_2'], u_parents=['U_Y'],
                                    f=lambda v, u: ((v["X_1"] and v["X_2"]) or u["U_Y"]))
    

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
        P_u_marginals={
            'U_X_1': P_X_1,
            'U_X_2': P_X_2,
            'U_Z_1': P_Z_1,
            'U_Z_2': P_Z_2,
            'U_Y': P_Y
        },
        F={
            'X_1': mechanism_X_1,
            'X_2': mechanism_X_2,
            'Z_1': mechanism_Z_1,
            'Z_2': mechanism_Z_2,
            'Y': mechanism_Y
        },
        seed=SEED
    )

    reward_node = 'Y'
    env = CausalBanditEnv(scm=scm, reward_node=reward_node, seed=SEED)
    print(f"Number of actions: {len(env.action_space)}")
    optimal_action, optimal_value = env.get_optimal(binary=True, discrete=True)  # Should be X_1=1, X_2=1
    print(f"optimal action is {optimal_action} with value {optimal_value}")

    G = env.scm.get_causal_diagram()

    agents = {
        "UCB": UCBAgent(reward_node=reward_node, arms=env.action_space, c=2),
        "POMIS-UCB": PomisUCBAgent(reward_node=reward_node, G=G, arms=env.action_space, c=2)
    }

    T= 1000  # number of steps in each run
    n = 1  # number of runs to average over


    regret = CumulativeRegret(optimal_expected_reward=optimal_value, T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    for name, agent in agents.items():
        print(f"Running agent: {name}")
        for _ in range(n):
            if _ % 100 == 0:
                print(f"  Run {_}/{n}")
            agent.reset()
            regret.reset()
            env.reset(seed=SEED + _)  # Ensure different randomness across runs
            for _ in range(T):
                action = agent.select_arm()
                print(f"Selected action: {action}")
                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                reward = observation[reward_node]
                regret.update(reward)
            
            averaged_regrets[name] += regret.get_regrets() / n

    plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")


if __name__ == "__main__":
    main()