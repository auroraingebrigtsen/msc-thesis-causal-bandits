"""
Implementation of the parallel bandit algorithm introduced in the paper:
Lattimore, F., Lattimore, T., & Reid, M. D. (2016).
Causal bandits: Learning good interventions via causal inference.
In Advances in Neural Information Processing Systems (Vol. 29).
"""

import numpy as np
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, NoisyORCPD

np.random.seed(42)
save_dir = 'preparations/plots'

# Variables
X = ["temp", "moist", "nutr"]
Y = "yield"

model = DiscreteBayesianNetwork([(x, Y) for x in X])

# Define the CPDs (Conditional Probability Distributions)
for var in X:
    prob = np.random.rand()
    cpd = TabularCPD( 
        variable=var, 
        variable_card=2, 
        values=[[prob], [1-prob]], 
        state_names={var: ["True", "False"]}
    )
    model.add_cpds(cpd)

prob_values = np.random.rand(len(X))

y_cpd = NoisyORCPD(
    variable=Y,
    prob_values=prob_values.tolist(),
    evidence=X,
)
model.add_cpds(y_cpd)

#  Validate
assert model.check_model()

# Get a daft object and save the figure
model_daft = model.to_daft()
model_daft.render()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_daft.savefig(os.path.join(save_dir, 'parallel_bandit_graph.png'))

# Define possible actions
actions = [(x, "True") for x in X] + [(x, "False") for x in X]


# PARALLEL BANDIT ALGORITHM
T = 1000   # number of iterations

# Simulate half of the iterations by purely observing
result = model.simulate(n_samples=T//2, do= {})

counts = {}  # count of how many times each node was True
q_hat = np.zeros(len(X), dtype=float)  # estimate of the probability of each node being True
probabilities = {}  # estimate of probability of each action

S = result.shape[0]
for idx, node in enumerate(X):
    vc = result[node].value_counts()
    count_true = vc.get("True", 0)

    counts[(node, "True")] = count_true
    counts[(node, "False")] = S - count_true

    q = count_true / S  #  2T_a / T = T_a / (T/2) = T_a / num_samples
    q_hat[idx] = q
    probabilities[(node, "True")] = q
    probabilities[(node, "False")] = 1 - q


# Estimate the reward for each action
avg_rewards = {}

for a in actions:  
    mu_hat = result.loc[result[a[0]] == a[1], Y].eq("True").mean()
    avg_rewards[a] = mu_hat

# Compute m_hat as a function of q_hat. m_hat is the threshold deciding whether we need to try an action more
# If m_hat is 2 we include all actions with <= 1/2 probability and if m_hat is N we include only actions with very small probability
def m(q_hat):
    # if  q = (1/2, 1/2, ..., 1/2) then m(q) = 2 and if q = (0, 0, ..., 0) then m(q) = N
    
    N = len(q_hat)
    for tau in range(2, N + 1): #  for tau in [2, N]
        # Compute I_tau = {i: min{q_i, 1-q_i} < 1/tau} 
        I_tau = np.sum(np.minimum(q_hat, 1 - q_hat) < 1.0 / tau)  # minimum produces a boolean mask of the elements, sum counts the instances
        # where the mask is True, so this line counts the number of elements satisfying the condition

        #  min{ tau: |I_tau| <= tau  }
        if I_tau <= tau:
            return tau  # We want the first tau that fits the criteria, as that one is the min
        
    return N

# Compute A = {a in Actions: prob < 1/m_hat}
m_hat = m(q_hat)
A = [a for a in actions if probabilities[a] <= 1 / m_hat]

# Calculate times to sample each low probability action
if len(A) == 0:
    # nothing to top-up; keep avg_rewards from observation-only phase
    T_A = 0
else:
    T_A = T // (2 * len(A))

#  Calculate new estimates for each low probability action
for var, val in A:  # iterate over the low probability actions
    result = model.simulate(n_samples=T_A, do={var: val}) # ex do={"temp" :True"}

    # calculate the new average reward
    new_avg= result.loc[result[var] == val, Y].eq("True").mean()

    #  re-estimate rewards
    old_avg=avg_rewards[(var, val)]
    avg_rewards[(var, val)] = ((old_avg*counts[(var, val)]) + (new_avg*T_A)) / (counts[(var, val)] + T_A)


print(f"Optimal action is {max(avg_rewards, key=avg_rewards.get)}")