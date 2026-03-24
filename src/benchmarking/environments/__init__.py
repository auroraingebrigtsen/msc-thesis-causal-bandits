from .markovian1 import build_markovian1
from .iv import build_iv

ENV_BUILDERS = {
    "markovian1": build_markovian1,
    "iv": build_iv,
}


def build_environment(params, seed):
    try:
        builder = ENV_BUILDERS[params["environment"]]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {params['environment']}") from e

    return builder(params, seed)