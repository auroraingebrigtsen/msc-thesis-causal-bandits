from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_markovian_3(params, seed, schedule=None):
    V = ['T', 'B', 'G', 'Y']
    U = ['U_T', 'U_B', 'U_G', 'U_Y']

    P_T = Bernoulli(p=params["p_t"])  
    P_B = Bernoulli(p=params["p_b"])  
    P_G = Bernoulli(p=params["p_g"])
    P_Y = Bernoulli(p=params["p_y"])

    mechanism_T = Mechanism(
        v_parents=[],
        u_parents=['U_T'],
        f=lambda _, u: int(u['U_T'])
    )
    mechanism_B = Mechanism(
        v_parents=[],
        u_parents=['U_B'],
        f=lambda _, u: int(u['U_B'])
    )
    # mechanism_G = Mechanism(
    #     v_parents=['B'],
    #     u_parents=['U_G'],
    #     f=lambda v, u: int(u['U_G']) | int(v['B'])
    # )
    mechanism_G = Mechanism(
        v_parents=[],
        u_parents=['U_G'],
        f=lambda _, u: int(u['U_G'])
    )
    mechanism_Y = Mechanism(
        v_parents=['T', 'B', 'G'],
        u_parents=['U_Y'],
        f=lambda v, u: v['T'] ^ v['B'] ^ u['U_Y']
    )
    
    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_T': P_T,
            'U_B': P_B,
            'U_G': P_G,
            'U_Y': P_Y
        },
        F={
            'T': mechanism_T,
            'B': mechanism_B,
            'G': mechanism_G,
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