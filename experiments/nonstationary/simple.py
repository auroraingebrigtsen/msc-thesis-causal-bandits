from cmab.algorithms.ucb.custom import MyFirstAgent
from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.algorithms.ucb import UCBAgent, SlidingWindowUCBAgent
from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.algorithms.ucb.custom import MyFirstAtomicAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np

### The setup from Task 1 in Structural Causal Bandits: Where to Intervene? ### 

def main():
    SEED = 42

    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=0.55)
    P_Z = Bernoulli(p=0.67)
    P_Y = Bernoulli(p=0.1)
    
    mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X'], f=lambda _, u: u['U_X'])
    mechanism_Z = CustomMechanism(v_parents=[], u_parents=['U_Z'], f=lambda _, u: u['U_Z'])
    mechanism_Y = CustomMechanism(v_parents=['X', 'Z'], u_parents=['U_Y'],
                                    f=lambda v, u: ((v["X"] or v["Z"]) ^ u["U_Y"]))
    

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y
        },
        F={
            'X': mechanism_X,
            'Z': mechanism_Z,
            'Y': mechanism_Y
        },
        seed=SEED
    )

    reward_node = 'Y'
    env = NSCausalBanditEnv(scm=scm, reward_node=reward_node, seed=SEED, atomic=True)
    print(f"Number of actions: {len(env.action_space)}")
    print(f"Action space: {env.action_space}")
    optimal_action, optimal_value = env.get_optimal(binary=True, discrete=True)  # Should be X=1
    print(f"optimal action is {optimal_action} with value {optimal_value}")

    G = env.scm.get_causal_diagram()

    agents = {
        #"UCB": UCBAgent(reward_node=reward_node, arms=env.action_space, c=2),
        "Custom-UCB": MyFirstAtomicAgent(reward_node=reward_node, G=G, arms=env.action_space, c=2)
    }

    T= 1000  # number of steps in each run
    n = 1  # number of runs to average over


    regret = CumulativeRegret(optimal_expected_reward=optimal_value, T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    for name, agent in agents.items():
        print(f"Running agent: {name}")
        for i in range(n):
            if i % 100 == 0:
                print(f"  Run {i}/{n}")

            agent.reset()
            regret.reset()
            # Use a different seed for SCM for each run. Use same seed for NS to have same change points across agents
            # If you want different change points across runs, use SEED + i for ns_seed
            env.reset(scm_seed=SEED+i, ns_seed=SEED)

            for _ in range(T):
                action = agent.select_arm()
                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                reward = observation[reward_node]
                regret.update(reward)
            
            averaged_regrets[name] += regret.get_regrets() / n

    plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")


if __name__ == "__main__":
    main()