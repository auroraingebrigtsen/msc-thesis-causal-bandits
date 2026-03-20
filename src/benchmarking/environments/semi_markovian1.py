from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule

def build_semi_markovian1(config):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y', 'U_XZ']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=0.9)
    P_Z = Bernoulli(p=0.75)
    P_Y = Bernoulli(p=0.2)
    P_XZ = Bernoulli(p=0.9)
    
    mechanism_X = CustomMechanism(v_parents=[], u_parents=['U_X', 'U_XZ'], 
                                  f=lambda _, u: u['U_X'] if u['U_XZ'] == 0 else 1 - u['U_X'])
    mechanism_Z = CustomMechanism(v_parents=['X'], u_parents=['U_Z', 'U_XZ'], 
                                  f=lambda v, u: (u['U_Z'] if u['U_XZ'] == 0 else 1 - u['U_Z'])  ^ v['X'])
    mechanism_Y = XORMechanism(v_parents=['Z'], u_parents=['U_Y'])

    scm = SCM(
        U=U,
        V=V,
        domains=domains,
        P_u_marginals={
            'U_X': P_X,
            'U_Z': P_Z,
            'U_Y': P_Y,
            'U_XZ': P_XZ
        },
        F={
            'X': mechanism_X,
            'Z': mechanism_Z,
            'Y': mechanism_Y
        },
        seed=config.seed
    )

    # 500: X, 1000: Y, 1500: Z, 2000: Z 
    reward_node = 'Y'
    schedule = ControlledSchedule(exogenous=['U_X', 'U_Y', 'U_X', 'U_Z'], new_params=[0.2, 0.9, 0.9, 0.1], every=500)

    return NSCausalBanditEnv(
        scm=scm,
        reward_node=reward_node,
        seed=config.seed,
        atomic=True,
        schedule=schedule,
        include_empty=False
    )