# experiments/environments/factory.py

from experiments.environments.markovian1 import build_simple_markovian1
from experiments.environments.markovian2 import build_simple_markovian2
from experiments.environments.semi_markovian1 import build_semi_markovian1

ENV_BUILDERS = {
    "simple_markovian1": build_simple_markovian1,
    "simple_markovian2": build_simple_markovian2,
    "semi_markovian1": build_semi_markovian1,
}


def build_environment(cfg):
    try:
        builder = ENV_BUILDERS[cfg.environment]
    except KeyError as e:
        raise ValueError(f"Unknown environment: {cfg.environment}") from e

    return builder(cfg.env_params)