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
            atomic=params["atomic"],
        )

    else:
        raise ValueError(f"Unknown agent: {name}")


def build_agents(agent_names: list[str], agent_params: dict, env):
    agents = {}
    for name in agent_names:
        agents[name] = build_agent(name, agent_params, env)
    return agents