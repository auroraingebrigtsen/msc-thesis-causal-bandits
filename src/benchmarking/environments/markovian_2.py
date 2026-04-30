from cmab.scm.distribution.uniform import Uniform
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import Mechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv
from benchmarking.utils import compute_change_points

def build_markovian_2(params, T, seed):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    P_X = Uniform(values=params["uniform"]["x_values"])
    P_Z = Uniform(values=params["uniform"]["z_values"])
    P_Y = Uniform(values=params["uniform"]["y_values"])

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
        f=lambda v, u: 2 * v['X'] - v['Z'] + u['U_Y'] >= 3
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
        include_empty=params["include_empty"],
        change_variables=params["change_params"]["variables"],
        updates=params["change_params"]["updates"],
        change_points=compute_change_points(T, params["change_params"])
    )