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
from cmab.algorithms.ucb.ph_ucb import PageHinkleyUCBAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np

def main():
    SEED = 42

    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=0.9)
    P_Z = Bernoulli(p=0.75)
    P_Y = Bernoulli(p=0.2)
    
    mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X'], f=lambda _, u: u['U_X'])
    mechanism_Z = CustomMechanism(v_parents=[], u_parents=['U_Z'], f=lambda _, u: u['U_Z'])
    mechanism_Y = XORMechanism(v_parents=['X', 'Z'], u_parents=['U_Y'])

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

    exogenous_to_shift = ['U_X', 'U_Z', 'U_Z', 'U_X', 'U_Z']
    new_probs = [0.6, 0.3, 0.8, 0.9, 0.4] 

    reward_node = 'Y'
    env = NSCausalBanditEnv(scm=scm, reward_node=reward_node, seed=SEED, atomic=True, exogenous_to_shift=exogenous_to_shift, new_probs=new_probs)
    print(f"Number of actions: {len(env.action_space)}")
    print(f"Action space: {env.action_space}")

    G = env.scm.get_causal_diagram()

    c = 2.0  # UCB exploration parameter
    delta = 0.01  # CPD tolerance parameter. 
    lambda_ = 2.0  # CPD threshold parameter
    min_samples_for_detection = 20  # Minimum samples before CPD starts detecting change points

    agents = {
        # Arm level CPD
        'PH-UCB': PageHinkleyUCBAgent(reward_node=reward_node, arms=env.action_space, c=c, delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection),
        'SW-UCB': SlidingWindowUCBAgent(reward_node=reward_node, arms=env.action_space, c=c, window_size=100),
        # Node level CPD
        #'Custom-UCB': MyFirstAtomicAgent(reward_node=reward_node, G=G, arms=env.action_space, c=c, delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection)
    }

    T= 1000  # number of steps in each run
    n = 1000  # number of runs to average over


    regret = CumulativeRegret(T=T)

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

            current_optimal, _ = env.get_optimal(binary=True, discrete=True)
            for _ in range(T):
                action = agent.select_arm()
                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                reward = observation[reward_node]
                optimal_arm, opt_exp_reward = env.get_optimal(binary=True, discrete=True)
                if optimal_arm != current_optimal:
                    print(f"  Optimal arm changed from {current_optimal} to {optimal_arm} at step {env._step}")
                    current_optimal = optimal_arm
                regret.update(reward, opt_exp_reward)
            
            averaged_regrets[name] += regret.get_regrets() / n

    plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")


if __name__ == "__main__":
    main()