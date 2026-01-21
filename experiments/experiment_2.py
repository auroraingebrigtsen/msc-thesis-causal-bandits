from cmab.scm.intervention_domain.interval import IntervalInterventionDomain
from cmab.scm.pmf.bernoulli import BernoulliPmf
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.algorithms.ucb import UCBAgent, SlidingWindowUCBAgent
from cmab.algorithms.ucb.pomis_ucb import PomisUCBAgent
from cmab.utils.plotting import  plot_regrets
from cmab.metrics.cumulative_regret import CumulativeRegret
import numpy as np

#  Recreating the setting from Structural Causal Bandits: Where to Intervene?
def main():
    SEED = 44

    V = frozenset({'S', 'T', 'W', 'Z', 'X', 'Y'})
    U = frozenset({'U_WX', 'U_ZY'})

    intervention_domains = {
        'S': IntervalInterventionDomain(0,1),
        'T': IntervalInterventionDomain(0,1),
        'W': IntervalInterventionDomain(0,1),
        'Z': IntervalInterventionDomain(0,1),
        'X': IntervalInterventionDomain(0,1),
        'Z': IntervalInterventionDomain(0,1),
        'Y': IntervalInterventionDomain(0,1)
    }

    P_X = BernoulliPmf(p=0.1)
    P_Z = BernoulliPmf(p=0.9)
    P_Y = BernoulliPmf(p=0.5)

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
    print(env.action_space)
    print(f"optimal action is {env.get_optimal_action()}")

    #ucb_agent = UCBAgent(n_arms=len(env.action_space), c=2)
    #ucb_agent = SlidingWindowUCBAgent(n_arms=len(env.action_space), c=2, window_size=100)
    G = env.scm.get_causal_diagram()
    ucb_agent = PomisUCBAgent(G=G, Y='Y', c=2)

    T= 1000  # number of steps in each run
    n = 1000  # number of runs to average over

    regret = CumulativeRegret(env=env, T=T) 

    averaged_regrets = np.zeros(T)
    for _ in range(n):
        ucb_agent.reset()
        regret.reset()
        env.reset()
        for _ in range(T):
            action_index = ucb_agent.select_arm()
            action =  env.action_space[action_index]
            _, reward, _, _, _ = env.step(action)
            ucb_agent._update(action_index, reward)
            regret.update(reward)
        
        averaged_regrets += regret.get_regrets() / n

    plot_regrets(regrets=[averaged_regrets], labels=["UCB Agent"], title="UCB Agent Averaged Cumulative Regret")


if __name__ == "__main__":
    main()