from cmab.algorithms.ucb import *
from cmab.algorithms.ucb.pht_vlr_ucb import PHTVlrUCBAgent
from cmab.algorithms.ucb.vlr_ucb_oracle import OracleVlrUCBAgent

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
            delta=params["arm_monitoring"]["delta"],
            lambda_=params["arm_monitoring"]["lambda"],
            min_samples_for_detection=params["arm_monitoring"]["min_samples_for_detection"],
            atomic=params["atomic"],
            reset_all=True,
            alpha=params["arm_monitoring"]["alpha"],
            seed=params["seed"]
        )

    elif name == "PHT-UCB-local":
        return PageHinkleyUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["arm_monitoring"]["delta"],
            lambda_=params["arm_monitoring"]["lambda"],
            min_samples_for_detection=params["arm_monitoring"]["min_samples_for_detection"],
            atomic=params["atomic"],
            reset_all=False,
            alpha=params["arm_monitoring"]["alpha"],
            seed=params["arm_monitoring"]["seed"]
        )

    elif name == "PHT-UCB-sr":
        return PHTSrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["variable_monitoring"]["delta"],
            lambda_=params["variable_monitoring"]["lambda"],
            min_samples_for_detection=params["variable_monitoring"]["min_samples_for_detection"],
            atomic=params["atomic"],
        )
    elif name == "PHT-VLR-UCB":
        return PHTVlrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            delta=params["arm_monitoring"]["delta"],
            lambda_=params["arm_monitoring"]["lambda"],
            min_samples_for_detection=params["arm_monitoring"]["min_samples_for_detection"],
            atomic=params["atomic"],
            alpha=params["arm_monitoring"]["alpha"],
            seed=params["arm_monitoring"]["seed"]
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
    elif name == "RBOCPD-UCB-local":
        return RBOCPDUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            gamma=params["gamma"],
            reset_all=False
        )
    elif name == "RBOCPD-UCB-sr":
        return RBOCPDSrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            gamma=params["gamma"],
        )
    elif name == "UCB-oracle-sr":
        return OracleSrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            changed_vars=env.change_variables,
            change_points=env.change_points
        )
    elif name == "UCB-oracle-local":
        return OracleUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            changed_vars=env.change_variables,
            change_points=env.change_points,
            reset_all=False
        )
    elif name == "UCB-oracle-global":
        return OracleUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            changed_vars=env.change_variables,
            change_points=env.change_points,
            reset_all=True
        )
    elif name == "UCB-oracle-vlr":
        return OracleVlrUCBAgent(
            reward_node=reward_node,
            G=G,
            arms=env.action_space,
            c=params["c"],
            atomic=params["atomic"],
            changed_vars=env.change_variables,
            change_points=env.change_points,
        )
    else:
        raise ValueError(f"Unknown agent: {name}")


def build_agents(agent_names: list[str], agent_params: dict, env):
    agents = {}
    for name in agent_names:
        agents[name] = build_agent(name, agent_params, env)
    return agents