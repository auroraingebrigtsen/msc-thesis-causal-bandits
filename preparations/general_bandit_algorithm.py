"""
Implementation of the general causal bandit algorithm (Algorithm 2) introduced in the paper:
Lattimore, F., Lattimore, T., & Reid, M. D. (2016).
Causal bandits: Learning good interventions via causal inference.
In Advances in Neural Information Processing Systems (Vol. 29).
"""

import numpy as np
import pandas as pd
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, NoisyORCPD

np.random.seed(42)
save_dir = 'preparations/plots'

# Variables
X = ["temp", "moist", "nutr"]
Y = "yield"

edges = [("temp", "moist"), ("moist", "nutr"), ("nutr", Y)] #  chain
model = DiscreteBayesianNetwork(edges)

# Define the CPDs (Conditional Probability Distributions)

# helper: Bernoulli CPD for root or 1-parent child
def bernoulli_cpd(var, p_true, parent=None):
    if parent is None:
        return TabularCPD(var, 2, [[p_true], [1 - p_true]],
                          state_names={var: ["True", "False"]})
    p_T, p_F = p_true  # P(var=True | parent=True/False)
    return TabularCPD(
        variable=var, variable_card=2,
        values=[[p_T, p_F], [1 - p_T, 1 - p_F]],
        evidence=[parent], evidence_card=[2],
        state_names={var: ["True", "False"], parent: ["True", "False"]}
    )


cpd_temp = bernoulli_cpd("temp", np.random.rand())
cpd_moist = bernoulli_cpd("moist", (np.random.rand(), np.random.rand()), parent="temp")
cpd_nutr = bernoulli_cpd("nutr", (np.random.rand(), np.random.rand()), parent="moist")
cpd_y  = bernoulli_cpd(Y, (np.random.rand(), np.random.rand()), parent="nutr")

model.add_cpds(cpd_temp, cpd_moist, cpd_nutr, cpd_y)

#  Validate
assert model.check_model()

# Get a daft object and save the figure
model_daft = model.to_daft()
model_daft.render()
os.makedirs(save_dir, exist_ok=True)
model_daft.savefig(os.path.join(save_dir, 'causal_bandit_graph.png'))

# Define possible actions
actions = [(x, "True") for x in X] + [(x, "False") for x in X]


# GENERAL CAUSAL BANDIT ALGORITHM
from pgmpy.inference import VariableElimination

T = 3 #  number of iterations

# Algorithm assumes we know P{Pa_Y | a}
# Pa_Y = {nutr} in this case, but could be several so we need to know the joint interventional distribution 
parents_Y = model.get_parents(Y)

# Joint distribution of Pa_Y under action a:  p_a[X] where X is a tuple of parent values, ex. P(nutr=true, moist=true| do(temp=false))
def joint_parent_dist_under_intervention(a:tuple): 
    do_model = model.do({a[0]: a[1]})
    inf = VariableElimination(do_model)
    q = inf.query(variables=parents_Y)   # joint over all parents

    # Build dict mapping tuple(parent values) -> prob
    p = {}

    # The shape of each parent's possible values
    possible_values = [q.state_names[parent] for parent in parents_Y] # e.g. [["True","False"], ["True","False"]]
    shape = [len(lst) for lst in possible_values]  # e.g. [2, 2]

    # Get all index combinations 
    it = np.ndindex(*shape)  # e.g. if shape = [2,2], then indices are (0,0), (0,1), (1,0), (1,1)
    for idx in it: 
        # Create a tuple with the actual variable names
        key = tuple(possible_values[i][idx[i]] for i in range(len(parents_Y))) 

        # Get the probability
        p[key] = float(q.values[idx])
    return p

p_tables = {a: joint_parent_dist_under_intervention(a) for a in actions}

# choose eta (distribution over actions),. Uniform for now, but paper suggests how this might be optimized
eta = np.ones(len(actions)) / len(actions) 
Q = {}
for s in next(iter(p_tables.values())).keys():
    Q[s] = sum(eta[j] * p_tables[actions[j]][s] for j in range(len(actions)))


#  m(eta) (difficulty) and truncation B_a from Theorem 3 
def m_of_eta(p_tables, Q):
    # m(eta) = max_a E_a[ P(PaY|a)/Q(PaY) ] = max_a sum_s p_a[s]^2 / Q[s]
    return max(sum((p_a[s] ** 2) / max(Q[s], 1e-12) for s in p_a) for p_a in p_tables.values())

m_eta = m_of_eta(p_tables, Q)
B = np.sqrt(m_eta * T / np.log(2 * T * len(actions)))  # scalar B_a (simple & effective)


# Importance ratio R_a(X_t)
def R_a(row: pd.Series, action):
    s = tuple(row[p] for p in parents_Y)
    num = p_tables[action][s]
    den = max(Q[s], 1e-12)
    return num / den

# Sampling loop
obs = []
for _ in range(T):
    idx = np.random.choice(len(actions), p=eta)
    var, val = actions[idx]
    sample = model.simulate(n_samples=1, do={var: val})
    obs.append(sample.iloc[0])  # store as Series

observations = pd.DataFrame(obs)

# Estimation: μ̂_a = (1/T) Σ Y_t * R_a(X_t) * 1{R_a≤B}
Ycol = Y  # clarity
estimates = np.zeros(len(actions))
for i, a in enumerate(actions):
    rs = observations.apply(lambda r: R_a(r, a), axis=1)
    mask = rs <= B
    estimates[i] = (observations.loc[mask, Ycol].eq("True").astype(float) * rs[mask]).sum() / T

best_idx = int(np.argmax(estimates))
print("Estimates of P(yield=True | do(a)) per action:")
for a, mu in zip(actions, estimates):
    print(f"  do({a[0]}={a[1]}): {mu:.4f}")
print(f"\nBest action: do({actions[best_idx][0]}={actions[best_idx][1]})  "
      f"with estimated reward {estimates[best_idx]:.4f}")