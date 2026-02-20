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
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule
from cmab.utils.plotting import  plot_regrets, plot_regrets_and_change_points, plot_detection_rate_heatmap
from cmab.metrics.dynamic_regret import DynamicRegret
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

    # Initial exogenous parameters (Segment 1)
    P_X = Bernoulli(p=0.9)   # p_X(0) = 0.9
    P_Z = Bernoulli(p=0.9)   # p_Z(0) = 0.9
    P_Y = Bernoulli(p=0.0)   # remove noise; keep clean XOR signal

    mechanism_X = CustomMechanism(
        v_parents=[],
        u_parents=['U_X'],
        f=lambda _, u: int(u['U_X'])
    )
    mechanism_Z = CustomMechanism(
        v_parents=[],
        u_parents=['U_Z'],
        f=lambda _, u: int(u['U_Z'])
    )
    mechanism_Y = CustomMechanism(
        v_parents=['X', 'Z'],
        u_parents=['U_Y'],
        # XOR. If you later want noise, set P_Y > 0.
        f=lambda v, u: int((v['X'] ^ v['Z']) ^ u['U_Y'])
    )

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

    # Change-points:
    # t=500:  U_X -> 0.1  (only Z-arms change; X-arms invariant)
    # t=1000: U_Z -> 0.1  (only X-arms change; Z-arms invariant)
    # t=1500: U_X -> 0.9  (only Z-arms change again)
    schedule = ControlledSchedule(
        exogenous=['U_X', 'U_Z', 'U_X'],
        new_params=[0.1, 0.1, 0.9],
        every=500
    )

    env = NSCausalBanditEnv(
        scm=scm,
        reward_node=reward_node,
        seed=SEED,
        atomic=True,
        shift_schedule=schedule,
        include_empty=False
    )

    print(f"Number of actions: {len(env.action_space)}")
    print(f"Action space: {env.action_space}")

    for action in env.action_space:
        expected_reward = scm.expected_value_binary(variable=reward_node, intervention_set=action)
        print(f"Expected reward for action {action}: {expected_reward:.4f}")

    G = env.scm.get_causal_diagram()

    c = 2.0  # UCB exploration parameter
    delta = 0.005  # CPD tolerance parameter. 
    lambda_ = 20.0  # CPD threshold parameter
    min_samples_for_detection = 30  # Minimum samples before CPD starts detecting change points

    agents = {
        # Arm level CPD
        #'UCB': UCBAgent(reward_node=reward_node, arms=env.action_space, c=c),
        'PH-UCB': PageHinkleyUCBAgent(reward_node=reward_node, arms=env.action_space, c=c, delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection, reset_all=True),
        'PH-UCB-arm': PageHinkleyUCBAgent(reward_node=reward_node, arms=env.action_space, c=c, delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection, reset_all=False),
        #'SW-UCB': SlidingWindowUCBAgent(reward_node=reward_node, arms=env.action_space, c=c, window_size=100),
        # Node level CPD
        'Custom-UCB': MyFirstAtomicAgent(reward_node=reward_node, G=G, arms=env.action_space, c=c, delta=delta, lambda_=lambda_, min_samples_for_detection=min_samples_for_detection)
    }

    T= 2000  # number of steps in each run
    n = 100  # number of runs to average over

    regret = DynamicRegret(T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    change_points = {
        name: {node: np.zeros(T, dtype=int) for node in G.nodes}
        for name in agents.keys()
    }
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
                optimal_arm, opt_exp_reward = env.get_optimal(binary=True, discrete=True)

                action = agent.select_arm()
                expected_reward = env.scm.expected_value_binary(variable=reward_node, intervention_set=action)

                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                expected_reward = env.scm.expected_value_binary(variable=reward_node, intervention_set=action)

                regret.update(expected_reward, opt_exp_reward)
            
            if hasattr(agent, 'change_points'):
                for node, cps in agent.change_points.items():
                    # cps is a list of length of T with values 0/1
                    change_points[name][node] += np.array(cps) 
            
            averaged_regrets[name] += regret.get_regrets() / n

    #plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")
    cps = schedule.get_change_points(T=T, rng=np.random.default_rng(SEED))
    plot_regrets_and_change_points(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret with Change Points", change_points=cps, T=T)
    # plot_detection_rate_heatmap(
    #     change_points=change_points,
    #     true_change_points=[200, 400, 600, 800, 900],
    #     title="Change detection rate by node",
    # )

if __name__ == "__main__":
    main()