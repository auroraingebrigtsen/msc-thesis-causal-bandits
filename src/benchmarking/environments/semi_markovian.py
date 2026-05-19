from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv
from benchmarking.utils import compute_change_points

def build_semi_markovian(params, T, seed):
    V = ['S', 'T', 'W', 'X', 'Z', 'Y']
    U = ['U_S', 'U_T', 'U_W', 'U_X', 'U_Z', 'U_Y' , 'U_WT', 'U_ZY']

    P_X = Bernoulli(p=params["p_x"])
    P_Z = Bernoulli(p=params["p_z"])
    P_Y = Bernoulli(p=params["p_y"])
    P_W = Bernoulli(p=params["p_w"])
    P_S = Bernoulli(p=params["p_s"])
    P_T = Bernoulli(p=params["p_t"])
    P_WT = Bernoulli(p=params["p_wt"])
    P_ZY = Bernoulli(p=params["p_zy"])

    mechanism_S = Mechanism(v_parents=[], u_parents=['U_S'], 
                            f=lambda v, u: u['U_S'])
    mechanism_T = Mechanism(v_parents=[], u_parents=['U_T', 'U_WT'], 
                            f=lambda v, u: u['U_T'] & u['U_WT'])
    mechanism_W = Mechanism(v_parents=['S'], u_parents=['U_W', 'U_WT'],
                            f=lambda v, u: u['U_W'] ^ v['S'] ^ u['U_WT'])
    mechanism_Z = Mechanism(v_parents=[], u_parents=['U_Z', 'U_ZY'],
                            f=lambda v, u: u['U_Z'] ^ u['U_ZY'])
    mechanism_X = Mechanism(v_parents=['T', 'Z'], u_parents=['U_X'],
                            f=lambda v, u: u['U_X'] & (v['T'] ^ v['Z']))
    mechanism_Y = Mechanism(v_parents=['T','W', 'X'], u_parents=['U_Y', 'U_ZY'],
                            f=lambda v, u: u['U_Y'] ^ u['U_ZY'] ^ v['T'] ^ v['W'] ^ v['X'])

    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_S': P_S,
            'U_T': P_T,
            'U_W': P_W,
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y,
            'U_WT': P_WT,
            'U_ZY': P_ZY,
        },
        F={
            'S': mechanism_S,
            'T': mechanism_T,
            'W': mechanism_W,
            'X': mechanism_X,
            'Z': mechanism_Z,
            'Y': mechanism_Y
        },
        seed=seed
    )
    return NSCausalBanditEnv(
        scm=scm,
        reward_node=params["reward_node"],
        seed=seed,
        atomic=params["atomic"],
        include_empty=params["include_empty"],
        change_variables=params["change_params"]["variables"],
        updates=params["change_params"]["updates"],
        change_points=compute_change_points(T, params["change_params"])
    )