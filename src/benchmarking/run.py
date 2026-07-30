from itertools import product
from joblib import Parallel, delayed
from benchmarking.plotting import plot_detected_nodes_heatmap, plot_regrets_and_change_points, plot_reset_rate_heatmap, plot_means
from cmab.utils.utils import compute_means_history
from cmab.metrics.dynamic_regret import DynamicRegret
from cmab.typing import Intervention
import numpy as np
from pathlib import Path
from .agent_factory import build_agents
from .environments import build_environment


def _run_single(name, cfg, seed, i, T, effective_action_space):
    """
    Run a single (agent, seed) trial in complete isolation.
    Rebuilds env and agent from scratch inside the worker so nothing
    is shared across processes -- avoids pickling issues and any
    cross-talk between parallel runs.
    """
    env = build_environment(cfg["env_params"], T, seed + i)  # fresh env, reseeded
    reward_node = env.reward_node
    agent = build_agents([name], cfg["agents_params"], env)[name]  # fresh agent for this env

    change_points = env.get_change_points()
    regret = DynamicRegret(T=T)

    optimal_arm, opt_exp_reward = env.get_optimal()
    for t in range(1, T + 1):
        action = agent.select_arm()
        _, observation, _, _, _ = env.step(action)
        agent._update(action, observation)
        expected_reward = env.scm.expected_value(variable=reward_node, intervention=action)
        if t in change_points:
            optimal_arm, opt_exp_reward = env.get_optimal()
        regret.update(expected_reward, opt_exp_reward)

    resat = {arm: np.zeros(T, dtype=int) for arm in effective_action_space}
    if hasattr(agent, 'resat_arms'):
        for arm, cps in agent.resat_arms.items():
            for cp in cps:
                resat[arm][cp - 1] += 1  # cp-1: agent uses 1-indexed steps, arrays are 0-indexed

    detected = {node: np.zeros(T, dtype=int) for node in env.scm.V}
    if hasattr(agent, 'detected_nodes'):
        for node, cps in agent.detected_nodes.items():
            for cp in cps:
                detected[node][cp - 1] += 1

    return name, i, regret.get_regrets(), resat, detected


def run(cfg, n_jobs=-1):
    seed = cfg["run"]["seed"]
    T = cfg["run"]["T"]  # number of steps in each run
    n = cfg["run"]["n"]  # number of runs to average over

    # Build once here just to inspect the environment / print setup info,
    # and to compute things (means_history, change_points, action spaces)
    # that are shared/reused for plotting, not for the actual trials.
    env = build_environment(cfg["env_params"], T, seed)
    reward_node = env.reward_node

    print(f"Running experiment: {cfg['name']}")
    print(f"Environment: {cfg['env_params']['environment']}")
    print(f"Environment has {len(env.action_space)} actions")

    agents = build_agents(cfg["agents"]["names"], cfg["agents_params"], env)

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

    # Flatten (agent_name, run_index) pairs so all cores stay busy
    # regardless of how many agents vs. how many runs there are.
    jobs = list(product(agents.keys(), range(n)))

    print(f"Dispatching {len(jobs)} trials across up to {n_jobs if n_jobs != -1 else 'all'} cores...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_single)(name, cfg, seed, i, T, effective_action_space)
        for name, i in jobs
    )

    # Aggregate results back into the original data structures
    all_regrets = {name: np.zeros((n, T)) for name in agents.keys()}
    resat_arms = {
        name: {arm: np.zeros(T, dtype=int) for arm in effective_action_space}
        for name in agents.keys()
    }
    averaged_detected_nodes = {node: np.zeros(T, dtype=int) for node in env.scm.V}

    for name, i, regrets, resat, detected in results:
        all_regrets[name][i] = regrets
        for arm, counts in resat.items():
            resat_arms[name][arm] += counts
        for node, counts in detected.items():
            averaged_detected_nodes[node] += counts

    regret_means = {
        name: regrets.mean(axis=0)
        for name, regrets in all_regrets.items()
    }

    regret_std = {
        name: regrets.std(axis=0, ddof=1)
        for name, regrets in all_regrets.items()
    }

    plot_detected_nodes_heatmap(
        detected_nodes=averaged_detected_nodes,
        agent_name="PHT-SR-UCB",
        save_path=path / "detected_nodes_heatmap.png"
    )

    plot_regrets_and_change_points(
        regrets=regret_means.values(),
        labels=regret_means.keys(),
        change_points=change_points,
        T=T,
        std_devs=regret_std,
        save_path=path / "regret.png"
    )

    for name, cps in resat_arms.items():
        plot_reset_rate_heatmap(
            reset_counts=cps,
            agent_name=name,
            save_path=path / f"reset_rate_{name}.png"
        )

    # Write the final cumulative regrets to a text file
    results_path = cfg.get('output', {}).get(
        'results_path',
        f"results/{cfg['env_params']['environment']}/{cfg['name']}/"
    )
    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(results_path / "results.txt", "w") as f:
        for name, regrets in regret_means.items():
            cumulative_regret = regrets[-1]
            f.write(f"{name}: {cumulative_regret:.3f} ± {regret_std[name][-1]:.3f}\n")