from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv

def build_semi_markovian(params, seed, schedule=None):
    V = ['S', 'T', 'W', 'X', 'Z', 'Y']
    U = ['U_S', 'U_T', 'U_W', 'U_X', 'U_Z', 'U_Y', 'U_WX' , 'U_ZY']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain(),
        'W': BinaryDomain(),
        'S': BinaryDomain(),
        'T': BinaryDomain()
    }

    P_X = Bernoulli(p=params["p_x"])
    P_Z = Bernoulli(p=params["p_z"])
    P_Y = Bernoulli(p=params["p_y"])
    P_W = Bernoulli(p=params["p_w"])
    P_S = Bernoulli(p=params["p_s"])
    P_T = Bernoulli(p=params["p_t"])
    P_WX = Bernoulli(p=params["p_wx"])
    P_ZY = Bernoulli(p=params["p_zy"])
    
    mechanism_S = CustomMechanism(v_parents=[], u_parents=['U_S'], 
                                  f=lambda v, u: u['U_S'])
    mechanism_T = CustomMechanism(v_parents=[], u_parents=['U_T'], 
                                  f=lambda v, u: u['U_T'])
    mechanism_W = CustomMechanism(v_parents=['S'], u_parents=['U_W', 'U_WX'],
                                    f=lambda v, u: u['U_W'] | (u['U_WX'] & v['S']))
    mechanism_Z = XORMechanism(v_parents=[], u_parents=['U_Z', 'U_ZY'])
    mechanism_X = CustomMechanism(v_parents=['T', 'Z'], u_parents=['U_X', 'U_WX'],
                                      f=lambda v, u: 1 ^ u['U_X'] ^ (u['U_WX'] | (v['T'] & v['Z'])))
    mechanism_Y = XORMechanism(v_parents=['T','W', 'X'], u_parents=['U_Y', 'U_ZY'])

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
        P_u_marginals={
            'U_S': P_S,
            'U_T': P_T,
            'U_W': P_W,
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y,
            'U_WX': P_WX,
            'U_ZY': P_ZY
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
        schedule=schedule,
        include_empty=params["include_empty"]
    )