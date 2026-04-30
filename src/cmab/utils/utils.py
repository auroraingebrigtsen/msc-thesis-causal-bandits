import copy

def compute_means_history(environment, T, effective_action_space=None):
    """
    Compute the means history for each arm in the environment over T time steps, taking into account the schedule's change events.

    Parameters:
    - environment: The environment object which contains the SCM and schedule.
    - T: Total number of time steps to compute the history for.
    - effective_action_space: Optional set of arms to compute the means for. If None, computes for all arms in the environment's action space.
    Returns:
    - A  dictionary mapping each arm to a list of means over time
    """
    env = copy.deepcopy(environment)  # Create a copy of the environment to avoid modifying the original SCM during mean computation
    env.action_space = effective_action_space if effective_action_space is not None else env.action_space
    means_history = dict((arm, []) for arm in env.action_space)
    change_points = env.schedule.get_change_points(T=T)
    for t in range(T):
        change_event = env.schedule.next(t=t)
        env.scm.apply_change_event(change_event)

        for arm in env.action_space:
            if t==0 or t in change_points:
                means_history[arm].append(env.scm.expected_value(variable=env.reward_node, intervention=arm))
            else:
                means_history[arm].append(means_history[arm][-1])
    return means_history