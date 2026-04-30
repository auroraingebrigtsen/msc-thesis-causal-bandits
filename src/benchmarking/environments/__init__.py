from .markovian import build_markovian
from .iv import build_iv
from .semi_markovian import build_semi_markovian
from .noncausal import build_noncausal
from .markovian_2 import build_markovian_2
from .markovian_3 import build_markovian_3

ENV_BUILDERS = {
    "markovian": build_markovian,
    "iv": build_iv,
    "semi_markovian": build_semi_markovian,
    "noncausal": build_noncausal,
    "markovian_2": build_markovian_2,
    "markovian_3": build_markovian_3
}


def build_environment(params, seed, schedule=None):
    try:
        builder = ENV_BUILDERS[params["environment"]]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {params['environment']}") from e

    return builder(params, seed, schedule=schedule)