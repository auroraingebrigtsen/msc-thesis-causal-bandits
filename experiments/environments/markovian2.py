from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.linear import LinearMechanism
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.mechanism.xor import XORMechanism
from cmab.scm.scm import SCM
from cmab.environments import CausalBanditEnv, NSCausalBanditEnv
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule

def build_simple_markovian1(config):
    V = ['X', 'Z', 'Y']
    U = ['U_X', 'U_Z', 'U_Y']

    domains = {
        'X': BinaryDomain(),
        'Z': BinaryDomain(),
        'Y': BinaryDomain()
    }

    P_X = Bernoulli(p=0.1)  
    P_Z = Bernoulli(p=0.7)  
    P_Y = Bernoulli(p=0.9)

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
    mechanism_Y = CustomMechanism(
        v_parents=['X', 'Z'],
        u_parents=['U_Y'],
        f=lambda v, u: int((v['X'] ^ v['Z']) ^ u['U_Y'])
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
        seed=config.seed
    )

    reward_node = 'Y'

    schedule = ControlledSchedule(
        exogenous=['U_X', 'U_X', 'U_X'],
        new_params=[0.9, 0.1, 0.9],
        every=500
    )

    return NSCausalBanditEnv(
        scm=scm,
        reward_node=reward_node,
        seed=config.seed,
        atomic=True,
        shift_schedule=schedule,
        include_empty=False
    )