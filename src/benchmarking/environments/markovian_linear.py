from cmab.scm.distribution.uniform import Uniform
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_markovian_linear(params, seed, schedule=None):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    P_X = Uniform(lower=params['x_lower'], upper=params['x_upper'])
    P_Z = Uniform(lower=params['z_lower'], upper=params['z_upper'])
    P_Y = Uniform(lower=params['y_lower'], upper=params['y_upper'])

    mechanism_X =  Mechanism(
        v_parents=[],
        u_parents=['U_X'],
        f=lambda _, u: u['U_X']
    )

    mechanism_Z = Mechanism(
        v_parents=[],
        u_parents=['U_Z'],
        f=lambda _, u: u['U_Z']
    )

    mechanism_Y = Mechanism(
        v_parents=['X', 'Z'],
        u_parents=['U_Y'],
        f=lambda v, u: v['X'] * 2 + v['Z'] * 1 + u['U_Y']
    )

    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y
        },
    )
            
    scm = SCM(
        U=U,
        V=V,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y
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