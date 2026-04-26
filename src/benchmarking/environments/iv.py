from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_iv(params, seed, schedule=None):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y', 'U_ZY']

    P_X = Bernoulli(p=params["p_x"])
    P_Z = Bernoulli(p=params["p_z"])
    P_Y = Bernoulli(p=params["p_y"])
    P_ZY = Bernoulli(p=params["p_zy"])

    if params["mechanism"] == "xor":
        mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X'], 
                                  f=lambda v, u: u['U_X'])
        mechanism_Z = XORMechanism(v_parents=['X'], u_parents=['U_Z', 'U_ZY'])
        mechanism_Y = XORMechanism(v_parents=['Z'], u_parents=['U_Y', 'U_ZY'])
    else:
        mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X'], 
                                  f=lambda v, u: u['U_X'])
        mechanism_Z = CustomMechanism(v_parents=['X'], u_parents=['U_Z', 'U_ZY'], 
                                      f=lambda v, u: u['U_Z'] | (u['U_ZY'] & v['X']))
        mechanism_Y = XORMechanism(v_parents=['Z'], u_parents=['U_Y', 'U_ZY'])

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
        schedule=schedule,
        include_empty=params["include_empty"]
    )