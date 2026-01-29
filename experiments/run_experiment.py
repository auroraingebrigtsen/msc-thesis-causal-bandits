from cmab.scm.domain.interval import IntervalDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.algorithms.ucb import UCBAgent, SlidingWindowUCBAgent
from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np

def main():
    SEED = 44

    V = frozenset({'X', 'Z', 'Y'})
    U = frozenset({'U_X', 'U_Z', 'U_Y'})

    intervention_domains = {
        'X': IntervalDomain(0,1),
        'Z': IntervalDomain(0,1),
        'Y': IntervalDomain(0,1)
    }

    P_X = Bernoulli(p=0.1)
    P_Z = Bernoulli(p=0.9)
    P_Y = Bernoulli(p=0.5)

    linear_mechanism_X = LinearMechanism(v_parents=[], u_parents=['U_X'], weights=[])
    linear_mechanism_Z = LinearMechanism(v_parents=['X'], u_parents=['U_Z'], weights=[0.8])
    linear_mechanism_Y = LinearMechanism(v_parents=['Z'], u_parents=['U_Y'], weights=[0.9])

    scm = SCM(
        U=U,
        V=V,
        intervention_domains=intervention_domains,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y
        },
        F={
            'X': linear_mechanism_X,
            'Z': linear_mechanism_Z,
            'Y': linear_mechanism_Y
        },
        seed=SEED
    )

    #env = NSCausalBanditEnv(scm=scm, reward_node='Y', prob_distribution_shift=0.0, max_delta=0.2, seed=SEED)
    env = CausalBanditEnv(scm=scm, reward_node='Y', seed=SEED)
    print(f"Number of actions: {len(env.action_space)}")
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
                action= agent.select_arm()
                _, reward, _, _, _ = env.step(action)
                agent._update(action, reward)
                regrets[name].update(reward)
        
        for name in agents.keys():
            averaged_regrets[name] += regrets[name].get_regrets() / n

    plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")


if __name__ == "__main__":
    main()