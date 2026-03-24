from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule

def build_iv(params, seed):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y', 'U_ZY']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=params["p_x"])
    P_Z = Bernoulli(p=params["p_z"])
    P_Y = Bernoulli(p=params["p_y"])
    P_ZY = Bernoulli(p=params["p_zy"])
    
    mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X'], 
                                  f=lambda v, u: u['U_X'])
    mechanism_Z = XORMechanism(v_parents=['X'], u_parents=['U_Z', 'U_ZY'])
    mechanism_Y = XORMechanism(v_parents=['Z'], u_parents=['U_Y', 'U_ZY'])

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
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

    # 500: X, 1000: Y, 1500: Z, 2000: Z 
    reward_node = 'Y'
    schedule = ControlledSchedule(
        exogenous=params["schedule"]["exogenous"], 
        new_params=params["schedule"]["new_params"], 
        every=params["schedule"]["every"]
        )

    return NSCausalBanditEnv(
        scm=scm,
        reward_node=reward_node,
        seed=seed,
        atomic=True,
        schedule=schedule,
        include_empty=False
    )