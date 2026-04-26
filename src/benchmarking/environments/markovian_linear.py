from cmab.scm.domain.interval import IntervalDomain
from cmab.scm.distribution.uniform import Uniform
from cmab.scm.mechanism import CustomMechanism, XORMechanism, LinearMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_markovian_linear(params, seed, schedule=None):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    domains = {
        'X': IntervalDomain(params['x_lower'], params['x_upper']),
        'Z': IntervalDomain(params['z_lower'], params['z_upper']),
        'Y': IntervalDomain(params['y_lower'], params['y_upper'])
    }

    P_X = Uniform(domain=domains['X'])
    P_Z = Uniform(domain=domains['Z'])
    P_Y = Uniform(domain=domains['Y'])

    mechanism_X = LinearMechanism(
        v_parents=[],
        u_parents=['U_X'],
        weights={
            'U_X': params['weights']['X']['U_X']
        },
    )

    mechanism_Z = LinearMechanism(
        v_parents=[],
        u_parents=['U_Z'],
        weights={
            'U_Z': params['weights']['Z']['U_Z']
        },
    )

    mechanism_Y = LinearMechanism(
        v_parents=['X', 'Z'],
        u_parents=['U_Y'],
        weights={
            'X': params['weights']['Y']['X'],
            'Z': params['weights']['Y']['Z'],
            'U_Y': params['weights']['Y']['U_Y'],
        },
    )
            
    scm = SCM(
        U=U,
        V=V,
        domains=domains,
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