from cmab.scm.domain.binary import BinaryDomain
from cmab.scm.distribution.bernoulli import Bernoulli
from cmab.scm.mechanism.custom import CustomMechanism
from cmab.scm.scm import SCM
from cmab.environments import NSCausalBanditEnv
from cmab.environments.ns.scheduling.controlled_schedule import ControlledSchedule

def build_markovian1(params, seed):
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
        seed=seed
    )

    schedule = ControlledSchedule(
        exogenous=params["schedule"]["exogenous"],
        new_params=params["schedule"]["new_params"],
        every=params["schedule"]["every"]
    )

    return NSCausalBanditEnv(
        scm=scm,
        reward_node=params["reward_node"],
        seed=seed,
        atomic=True,
        shift_schedule=schedule,
        include_empty=False
    )