from .markovian1 import build_markovian1
from .markovian2 import build_markovian2
from .semi_markovian1 import build_semi_markovian1

ENV_BUILDERS = {
    "markovian1": build_markovian1,
    "markovian2": build_markovian2,
    "semi_markovian1": build_semi_markovian1,
}


def build_environment(params, seed):
    try:
        builder = ENV_BUILDERS[params["environment"]]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {params['environment']}") from e

    return builder(params, seed)