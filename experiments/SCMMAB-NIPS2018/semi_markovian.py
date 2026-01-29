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

### The setup from Task 3 in Structural Causal Bandits: Where to Intervene? ### 

def main():
    SEED = 42

    V = frozenset({'S', 'T', 'W', 'Z', 'X', 'Y'})
    U = frozenset({'U_S', 'U_T', 'U_W', 'U_X', 'U_Y', 'U_Z', 'U_WX', 'U_ZY'})

    domains = {
        'S': BinaryDomain(),
        'T': BinaryDomain(),
        'W': BinaryDomain(),
        'Z': BinaryDomain(),
        'X': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_U_S = Bernoulli(p=0.45)
    P_U_T = Bernoulli(p=0.81)
    P_U_W = Bernoulli(p=0.07)
    P_U_X = Bernoulli(p=0.06)
    P_U_Y = Bernoulli(p=0.06)
    P_U_Z = Bernoulli(p=0.05)
    P_U_WX = Bernoulli(p=0.51)
    P_U_ZY = Bernoulli(p=0.54)
    
    mechanism_S = CustomMechanism(v_parents=[], u_parents=['U_S'], f=lambda _, u: u['U_S'])
    mechanism_T = CustomMechanism(v_parents=[], u_parents=['U_T'], f=lambda _, u: u['U_T'])
    mechanism_W = XORMechanism(v_parents=['S'], u_parents=['U_W', 'U_WX'])
    mechanism_Z = XORMechanism(v_parents=[], u_parents=['U_Z', 'U_ZY'])
    mechanism_X = CustomMechanism(v_parents=['T', 'Z'], u_parents=['U_X', 'U_WX'],
                                    f=lambda v, u: 1 ^ v['T'] ^ v['Z'] ^ u['U_X'] ^ u['U_WX'])
    mechanism_Y = XORMechanism(v_parents=['T', 'W', 'X'], u_parents=['U_Y', 'U_ZY'])
    

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
        P_u_marginals={
            'U_S': P_U_S,
            'U_T': P_U_T,
            'U_W': P_U_W,
            'U_X': P_U_X,
            'U_Y': P_U_Y,
            'U_Z': P_U_Z,
            'U_WX': P_U_WX,
            'U_ZY': P_U_ZY
        },
        F={
            'S': mechanism_S,
            'T': mechanism_T,
            'W': mechanism_W,
            'Z': mechanism_Z,
            'X': mechanism_X,
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
            env.reset()
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