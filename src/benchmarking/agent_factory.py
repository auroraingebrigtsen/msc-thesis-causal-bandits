from cmab.algorithms.ucb import *

def build_agent(name: str, params: dict, env):
    G = env.scm.get_causal_diagram()
    reward_node = env.reward_node

    if name == "UCB":
        return UCBAgent(
            reward_node=reward_node,
            arms=env.action_space,
            c=params["c"],
        )
    
    elif name == "POMIS-UCB":
        return PomisUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
        )

    elif name == "PHT-UCB-global":
        return PageHinkleyUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["delta"],
            lambda_=params["lambda"],
            min_samples_for_detection=params["min_samples_for_detection"],
            atomic=params["atomic"],
            reset_all=True,
        )

    elif name == "PHT-UCB-arm":
        return PageHinkleyUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["delta"],
            lambda_=params["lambda"],
            min_samples_for_detection=params["min_samples_for_detection"],
            atomic=params["atomic"],
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
    elif name == "RBOCPD-UCB-global":
        return RBOCPDUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            gamma=params["gamma"],
            reset_all=True
        )
    elif name == "RBOCPD-UCB-arm":
        return RBOCPDUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            gamma=params["gamma"],
            reset_all=False
        )
    elif name == "RBOCPD-SR-UCB":
        return RBOCPDSrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            gamma=params["gamma"],
        )
    else:
        raise ValueError(f"Unknown agent: {name}")


def build_agents(agent_names: list[str], agent_params: dict, env):
    agents = {}
    for name in agent_names:
        agents[name] = build_agent(name, agent_params, env)
    return agents