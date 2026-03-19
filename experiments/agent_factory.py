from cmab.algorithms.ucb import UCBAgent
from cmab.algorithms.ucb.sr_ucb import SrUCBAgent
from cmab.algorithms.ucb.ph_ucb import PageHinkleyUCBAgent


def build_agent(name: str, params: dict, env):
    G = env.scm.get_causal_diagram()
    reward_node = env.reward_node

    if name == "UCB":
        return UCBAgent(
            reward_node=reward_node,
            arms=env.action_space,
            c=params["c"],
        )

    elif name == "PH-UCB":
        return PageHinkleyUCBAgent(
            reward_node=reward_node,
            arms=env.action_space,
            c=params["c"],
            delta=params["delta"],
            lambda_=params["lambda"],
            min_samples_for_detection=params["min_samples_for_detection"],
            reset_all=True,
        )

    elif name == "PH-UCB-arm":
        return PageHinkleyUCBAgent(
            reward_node=reward_node,
            arms=env.action_space,
            c=params["c"],
            delta=params["delta"],
            lambda_=params["lambda"],
            min_samples_for_detection=params["min_samples_for_detection"],
            reset_all=False,
        )

    elif name == "SR-UCB":
        return SrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["delta"],
            lambda_=params["lambda"],
            min_samples_for_detection=params["min_samples_for_detection"],
        )

    else:
        raise ValueError(f"Unknown agent: {name}")


def build_agents(cfg, env):
    agents = {}
    for name, params in cfg.agents.items():
        agents[name] = build_agent(name, cfg.agent_params, env)
    return agents