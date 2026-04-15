import copy

def compute_means_history(environment, T):
    """
    Compute the means history for each arm, inserting NaN values at the specified breakpoints.

    Parameters:
    - environment: The environment object which contains the SCM and schedule.
    - T: Total number of time steps to compute the history for.
    Returns:
    - A  dictionary mapping each arm to a list of means over time
    """
    env = copy.deepcopy(environment)  # Create a copy of the environment to avoid modifying the original SCM during mean computation
    means_history = dict((arm, []) for arm in env.action_space)
    change_points = env.schedule.get_change_points(T=T)
    for t in range(T):
        if t in change_points:
            env.scm.apply_shift(env.schedule.next(t=t))
        for arm in env.action_space:
            if t==0 or t in change_points:
                means_history[arm].append(env.scm.expected_value_binary(variable=env.reward_node, intervention=arm))
            else:
                means_history[arm].append(means_history[arm][-1])
    return means_history