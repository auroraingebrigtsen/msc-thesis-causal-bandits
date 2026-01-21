from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.pmf.bernoulli import BernoulliPmf
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

    P_X_1 = BernoulliPmf(p=0.54)
    P_X_2 = BernoulliPmf(p=0.67)
    P_Z_1 = BernoulliPmf(p=0.54)
    P_Z_2 = BernoulliPmf(p=0.44)
    P_Y = BernoulliPmf(p=0.58)

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

    #env = NSCausalBanditEnv(scm=scm, reward_node='Y', prob_distribution_shift=0.0, max_delta=0.2, seed=SEED)
    env = CausalBanditEnv(scm=scm, reward_node='Y', seed=SEED)
    print(env.action_space)
    print(f"optimal action is {env.get_optimal_action()}")

    G = env.scm.get_causal_diagram()

    agents = {
        "UCB": UCBAgent(n_arms=len(env.action_space), c=2),
        "SW-UCB": SlidingWindowUCBAgent(n_arms=len(env.action_space), c=2, window_size=100),
        "POMIS-UCB": PomisUCBAgent(G=G, Y='Y', c=2)
    }

    T= 1000  # number of steps in each run
    n = 1000  # number of runs to average over


    regrets = {name: CumulativeRegret(env=env, T=T) for name in agents.keys()}

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    for _ in range(n):
        for agent in agents.values():
            agent.reset()
        for regret in regrets.values():
            regret.reset()
        env.reset()
        for _ in range(T):
            for name, agent in agents.items():
                action_index = agent.select_arm()
                action =  env.action_space[action_index]
                _, reward, _, _, _ = env.step(action)
                agent._update(action_index, reward)
                regrets[name].update(reward)
        
        for name in agents.keys():
            averaged_regrets[name] += regrets[name].get_regrets() / n

    plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")


if __name__ == "__main__":
    main()