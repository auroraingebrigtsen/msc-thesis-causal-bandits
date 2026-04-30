from benchmarking.plotting import  plot_regrets_and_change_points, plot_reset_rate_heatmap, plot_historical_means
from cmab.utils.utils import compute_means_history
from cmab.metrics.dynamic_regret import DynamicRegret
from cmab.typing import Intervention
import numpy as np
from pathlib import Path
from .agent_factory import build_agents
from .environments import build_environment
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule
from cmab.environments.ns.scheduling.stationary_schedule import StationarySchedule

def run(cfg):
    seed = cfg["run"]["seed"]
    env_params = cfg["env_params"]

    if env_params["schedule"]["type"] == "controlled_schedule":
        schedule = ControlledSchedule(
            variables=env_params["schedule"].get("variables", []),
            update=env_params["schedule"].get("update", []),
            every=env_params["schedule"].get("every", 0)
        )
    else:
        schedule = StationarySchedule()

    env = build_environment(cfg["env_params"], seed, schedule)
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

    T= cfg["run"]["T"]  # number of steps in each run
    n = cfg["run"]["n"]  # number of runs to average over

    output_path = cfg['output'].get(
    'output_path',
    f"plots/{cfg['env_params']['environment']}/{cfg['name']}/"
    )

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    change_points = env.schedule.get_change_points(T=T)
    means_history = compute_means_history(env, T=T, effective_action_space=effective_action_space)
    plot_historical_means(
        means_history=means_history,
        change_points=change_points,
        save_path=path / "historical_means.png"
    )

    regret = DynamicRegret(T=T)

    averaged_regrets = {name: np.zeros(T) for name in agents.keys()}
    resat_arms = {
        name: {arm: np.zeros(T, dtype=int) for arm in effective_action_space} 
        for name in agents.keys()
    }

    for name, agent in agents.items():
        print(f"Running agent: {name}")
        optimal_arm, opt_exp_reward = env.get_optimal()
        for i in range(n):
            if i % 10 == 0:
                print(f"  Run {i}/{n}")

            agent.reset()
            regret.reset()
            # Use a different seed for SCM for each run. Use same seed for NS to have same change points across agents
            # If you want different change points across runs, use SEED + i for ns_seed
            env.reset(scm_seed=seed+i, ns_seed=seed)
            for t in range(T):
                if t in change_points:
                    optimal_arm, opt_exp_reward = env.get_optimal()
                    
                action = agent.select_arm()

                _, observation, _, _, _ = env.step(action)
                agent._update(action, observation)
                expected_reward = env.scm.expected_value(variable=reward_node, intervention=action)
                regret.update(expected_reward, opt_exp_reward)
            
            if hasattr(agent, 'resat_arms'):
                for arm, cps in agent.resat_arms.items():
                    for cp in cps:
                        resat_arms[name][arm][cp-1] += 1  # cp-1 because time steps are 1-indexed in the agent but we want 0-indexed for the array

            averaged_regrets[name] += regret.get_regrets() / n

    plot_regrets_and_change_points(
        regrets=averaged_regrets.values(),
        labels=averaged_regrets.keys(),
        title="Averaged Cumulative Regret",
        change_points=change_points,
        T=T,
        save_path=path / "regret.png"
    )

    for name, cps in resat_arms.items():
        plot_reset_rate_heatmap(
            reset_counts=cps,
            title=f"Reset rate by arm for agent {name}",
            save_path=path / f"reset_rate_{name}.png"
        )