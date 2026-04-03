from .markovian import build_markovian
from .iv import build_iv
from .semi_markovian import build_semi_markovian

ENV_BUILDERS = {
    "markovian": build_markovian,
    "iv": build_iv,
    "semi_markovian": build_semi_markovian
}


def build_environment(params, seed):
    try:
        builder = ENV_BUILDERS[params["environment"]]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {params['environment']}") from e

    return builder(params, seed)