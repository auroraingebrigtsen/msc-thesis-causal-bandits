from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_noncausal(params, seed, schedule=None):
    V = ['X', 'Y']
    U = ['U_0', 'U_1']

    P_0 = Bernoulli(p=params["p_0"])  
    P_1 = Bernoulli(p=params["p_1"])

    mechanism_X = Mechanism(
        v_parents=[],
        u_parents=[],
        f=lambda _, u: 0
    )
    mechanism_Y = Mechanism(
        v_parents=['X'],
        u_parents=['U_0', 'U_1'],
        f=lambda v, u: int(u['U_0'])  if int(v['X']) == 0 else int(u['U_1'])
    )
        
    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_0': P_0,
            'U_1': P_1
        },
        F={
            'X': mechanism_X,
            'Y': mechanism_Y
        },
        seed=seed
    )

    return NSCausalBanditEnv(
        scm=scm,
        reward_node=params["reward_node"],
        seed=seed,
        atomic=params["atomic"],
        schedule=schedule,
        include_empty=params["include_empty"]
    )