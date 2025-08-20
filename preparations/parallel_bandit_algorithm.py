"""
Implementation of the parallel bandit algorithm introduced in the paper:
Lattimore, F., Lattimore, T., & Reid, M. D. (2016).
Causal bandits: Learning good interventions via causal inference.
In Advances in Neural Information Processing Systems (Vol. 29).
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, NoisyORCPD

np.random.seed(42)

# DEFINE THE SCM

# Variables
X = ["temp", "moist", "nutr"]
Y = "yield"

model = DiscreteBayesianNetwork([(x, Y) for x in X])

# Define the CPDs (Conditional Probability Distributions)
for var in X:
    prob = np.random.rand()
    print(f"Probability of {var} being True: {prob:.2f}")
    cpd = TabularCPD( 
        variable=var, 
        variable_card=2, 
        values=[[prob], [1-prob]], 
        state_names={var: ["True", "False"]}
    )
    model.add_cpds(cpd)

prob_values = np.random.rand(len(X))
print(f"Probabilities for {Y} given {X} is True: {prob_values}")
y_cpd = NoisyORCPD(
    variable=Y,
    prob_values=prob_values.tolist(),
    evidence=X,
)
model.add_cpds(y_cpd)

#  Validate
assert model.check_model()

# Get a daft object
model_daft = model.to_daft()
model_daft.render()
model_daft.savefig('test.png')

# Define possible actions
actions = [(x, "True") for x in X] + [(x, "False") for x in X]
print(f"Actions: {actions}")


# PARALLEL BANDIT ALGORITHM
T = 1000   # number of iterations

# Simulate half of the iterations by purely observing
result = model.simulate(n_samples=T//2, do= {})
print(result.head())
print(result.shape)

counts = {}  # Count how many times each node was True
q_hat = np.array([0.0] * len(X))  # Estimate of the probability of each node being True
probabilities = {}  # Estimate of the probability of each action being True

for idx, node in enumerate(X):
    count = result[node].value_counts()
    count_true = count.get("True", 0)
    count_false = count.get("False", 0)
    print(f"Counts for {node}:")
    print(count)

    counts[(node, "True")] = count_true
    counts[(node, "False")] = count_false

    probabilities[(node, "True")] = 2 *  count_true / result.shape[0]
    probabilities[(node, "False")] = 2 * count_false / result.shape[0]

    q_hat[idx] = 2 * count_true / result.shape[0]


# Estimate reward for each action
avg_rewards = {}

for a in actions:  
    mu_hat = result.loc[result[a[0]] == a[1], Y].eq("True").mean()
    print(f"Estimated reward for {a[0]}={a[1]}: {mu_hat}")

    avg_rewards[a] = mu_hat

# Compute m_hat as a function of q_hat. m_hat is the threshold deciding whether we need to try an action more
# If m_hat is 2 we include all actions with <= 1/2 probability and if m_hat is N we include only actions with very small probability
def m(q_hat):
    # if  q = (1/2, 1/2, ..., 1/2) then m(q) = 2 and if q = (0, 0, ..., 0) then m(q) = N

    m_hat = np.inf  # Initialize m_hat to infinity
    for tau in range(2, len(X) + 1): #  for tau in [2, N]
        # Compute I_tau = {i: min{q_i, 1-q_i} < 1/tau} 
        temp = np.flatnonzero(np.minimum(q_hat, 1 - q_hat) < 1.0 / tau).tolist()

        #  min{ tau: |I_tau| <= tau  }
        if len(temp) <= tau and tau < m_hat:
            m_hat = tau

    return m_hat if m_hat is not np.inf else 1

# Compute A = {a in Actions: prob < 1/m_hat}
m_hat = m(q_hat)
A = [a for a in actions if probabilities[a] <= 1 / m_hat]

print(f"Actions with probability < 1/m_hat: {A}")

T_A = T // (2 * len(A))  # times to sample each low probability action

for var, val in A:  # iterate over the low probability actions
    result = model.simulate(n_samples=T_A, do={var: val}) # ex do={"temp" :True"}
    print(result.head())
    print(result.shape)

    # calculate the new average reward
    new_avg= result.loc[result[var] == val, Y].eq("True").mean()
    print(f"Estimated new reward for {var}={val}: {new_avg}")

    #  re-estimate rewards
    old_avg=avg_rewards[(var, val)]
    avg_rewards[(var, val)] = ((old_avg*counts[(var, val)]) + (new_avg*T_A)) / (counts[(var, val)] + T_A)


print(f"Average rewards after re-estimation: {avg_rewards}")
print(f"Optimal action is {max(avg_rewards, key=avg_rewards.get)}")