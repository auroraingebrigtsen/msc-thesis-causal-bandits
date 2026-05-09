
from benchmarking.plotting import  plot_regrets_and_change_points, plot_reset_rate_heatmap, plot_means
from cmab.utils.utils import compute_means_history
from cmab.metrics.dynamic_regret import DynamicRegret
from cmab.typing import Intervention
import numpy as np
from pathlib import Path
from .agent_factory import build_agents
from .environments import build_environment


def run(cfg):
    seed = cfg["run"]["seed"]
    T= cfg["run"]["T"]  # number of steps in each run
    n = cfg["run"]["n"]  # number of runs to average over

    env = build_environment(cfg["env_params"], T,  seed)
    reward_node = env.reward_node

    print(f"Running experiment: {cfg['name']}")
    print(f"Environment: {cfg['env_params']['environment']}")
    print(f"Environment has {len(env.action_space)} actions")

    agents = build_agents(cfg["agents"]["names"],  cfg["agents_params"], env)

    action_space: set[Intervention] = set(env.action_space)
    agent_action_space = {arm for agent in agents.values() for arm in agent.arms}
    effective_action_space = action_space & agent_action_space

    for action in effective_action_space:
        print(
            f"Arm: {action}, Expected reward: "
            f"{env.scm.expected_value(variable=reward_node, intervention=action)}"
        )

    plots_path = cfg.get('output', {}).get(
    'plots_path',
    f"plots/{cfg['env_params']['environment']}/{cfg['name']}/"
    )
    path = Path(plots_path)
    path.mkdir(parents=True, exist_ok=True)
    change_points = env.get_change_points()
    means_history = compute_means_history(env, T=T, effective_action_space=effective_action_space)

    plot_means(
        means_history=means_history,
        change_points=change_points,
        save_path=path / "means.png"
    )
    regret = DynamicRegret(T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    resat_arms = {
        name: {arm: np.zeros(T, dtype=int) for arm in effective_action_space} 
        for name in agents.keys()
    }

    for name, agent in agents.items():
        print(f"Running agent: {name}")
        for i in range(n):
            if i % 10 == 0:
                print(f"  Run {i}/{n}")

            agent.reset()
            regret.reset()
            env.reset(seed=seed + i)  # Use a different seed for the SCM at each run 
            
            optimal_arm, opt_exp_reward = env.get_optimal()
            for t in range(T):
                action = agent.select_arm()
                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                expected_reward = env.scm.expected_value(variable=reward_node, intervention=action)
                if t in change_points:
                    optimal_arm, opt_exp_reward = env.get_optimal()
                regret.update(expected_reward, opt_exp_reward)
            
            if hasattr(agent, 'resat_arms'):
                for arm, cps in agent.resat_arms.items():
                    for cp in cps:
                        resat_arms[name][arm][cp-1] += 1  # cp-1 because time steps are 1-indexed in the agent but we want 0-indexed for the array

            averaged_regrets[name] += regret.get_regrets() / n

    plot_regrets_and_change_points(
        regrets=averaged_regrets.values(),
        labels=averaged_regrets.keys(),
        change_points=change_points,
        T=T,
        save_path=path / "regret.png"
    )

    for name, cps in resat_arms.items():
        plot_reset_rate_heatmap(
            reset_counts=cps,
            agent_name=name,
            save_path=path / f"reset_rate_{name}.png"
        )

    # WRite the final cumulative regrets to a text file
    results_path = cfg.get('output', {}).get(
        'results_path',
        f"results/{cfg['env_params']['environment']}/{cfg['name']}/"
    )
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "regrets.txt", "w") as f:
        for name, regrets in averaged_regrets.items():
            cumulative_regret =regrets[-1]
            f.write(f"{name}: {cumulative_regret}\n")