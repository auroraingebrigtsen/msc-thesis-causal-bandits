from .markovian import build_markovian
from .iv import build_iv
from .semi_markovian import build_semi_markovian
from .noncausal import build_noncausal

ENV_BUILDERS = {
    "markovian": build_markovian,
    "iv": build_iv,
    "semi_markovian": build_semi_markovian,
    "noncausal": build_noncausal
}


def build_environment(params, seed, schedule=None):
    try:
        builder = ENV_BUILDERS[params["environment"]]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {params['environment']}") from e

    return builder(params, seed, schedule=schedule)