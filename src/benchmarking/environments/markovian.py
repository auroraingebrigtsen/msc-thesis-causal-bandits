from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism import CustomMechanism, XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_markovian(params, seed, schedule=None):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=params["p_x"])  
    P_Z = Bernoulli(p=params["p_z"])  
    P_Y = Bernoulli(p=params["p_y"])

    mechanism_X = CustomMechanism(
        v_parents=[],
        u_parents=['U_X'],
        f=lambda _, u: int(u['U_X'])
    )
    mechanism_Z = CustomMechanism(
        v_parents=[],
        u_parents=['U_Z'],
        f=lambda _, u: int(u['U_Z'])
    )
    mechanism_Y = XORMechanism(
        v_parents=['X', 'Z'],
        u_parents=['U_Y']
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