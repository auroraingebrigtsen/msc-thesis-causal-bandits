from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv
from benchmarking.utils import compute_change_points

def build_iv(params, T, seed):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y', 'U_ZY']

    P_X = Bernoulli(p=params["p_x"])
    P_Z = Bernoulli(p=params["p_z"])
    P_Y = Bernoulli(p=params["p_y"])
    P_ZY = Bernoulli(p=params["p_zy"])

    mechanism_X = Mechanism(v_parents=[], u_parents=['U_X'], 
                                f=lambda v, u: u['U_X'])
    mechanism_Z = Mechanism(v_parents=['X'], u_parents=['U_Z', 'U_ZY'], 
                                    f=lambda v, u: v['X'] ^ u['U_ZY'] & u['U_Z'])
    mechanism_Y = Mechanism(v_parents=['Z'], u_parents=['U_Y', 'U_ZY'], f=lambda v, u: v['Z'] ^  (u['U_Y'] & u['U_ZY']))

    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y,
            'U_ZY': P_ZY
        },
        F={
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