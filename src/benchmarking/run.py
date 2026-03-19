from cmab.utils.plotting import  plot_regrets_and_change_points, plot_reset_rate_heatmap
from cmab.metrics.dynamic_regret import DynamicRegret
import numpy as np
from .agent_factory import build_agents
from .environments import build_environment

def run(cfg):
    seed = cfg["run"]["seed"]
    env = build_environment(cfg["env_params"], seed)
    reward_node = env.reward_node

    print(f"Number of actions: {len(env.action_space)}")
    print(f"Action space: {env.action_space}")

    for action in env.action_space:
        expected_reward = env.scm.expected_value_binary(variable=reward_node, intervention=action)
        print(f"Expected reward for action {action}: {expected_reward:.4f}")

    agents = build_agents(cfg["agents"]["names"],  cfg["agents_params"], env)

    T= cfg["run"]["T"]  # number of steps in each run
    n = cfg["run"]["n"]  # number of runs to average over

    regret = DynamicRegret(T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    resat_arms = {
        name: {arm: np.zeros(T, dtype=int) for arm in env.action_space} 
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
            env.reset(scm_seed=seed+i, ns_seed=seed)

            for _ in range(T):
                optimal_arm, opt_exp_reward = env.get_optimal(binary=True)

                action = agent.select_arm()
                expected_reward = env.scm.expected_value_binary(variable=reward_node, intervention=action)

                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                expected_reward = env.scm.expected_value_binary(variable=reward_node, intervention=action)

                regret.update(expected_reward, opt_exp_reward)
            
            if hasattr(agent, 'resat_arms'):
                for arm, cps in agent.resat_arms.items():
                    for cp in cps:
                        resat_arms[name][arm][cp-1] += 1  # cp-1 because time steps are 1-indexed in the agent but we want 0-indexed for the array

            averaged_regrets[name] += regret.get_regrets() / n

    #plot_regrets(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret")
    cps = env.schedule.get_change_points(T=T, rng=np.random.default_rng(seed))
    plot_regrets_and_change_points(regrets=averaged_regrets.values(), labels=averaged_regrets.keys(), title="Averaged Cumulative Regret with Change Points", 
                                   change_points=cps, T=T, save_path=cfg["output"]["plot_regret_path"])
    for name, cps in resat_arms.items():
        plot_reset_rate_heatmap(reset_counts=cps,title=f"Reset rate by arm for agent {name}", 
                                save_path=f"plots/{cfg['output']['plot_reset_heatmap_prefix']}_{name}.png")