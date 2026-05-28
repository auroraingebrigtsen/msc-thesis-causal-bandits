import tomllib
from pathlib import Path
from benchmarking.run import run

def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)

configs = [
    #"configs/markovian/large_changes.toml",
    #"configs/markovian/large_changes_oracles.toml",
    #"configs/markovian/medium_changes.toml",
    #"configs/markovian/medium_changes_oracles.toml",
    #"configs/iv/iv_mechanism_changes.toml",
    "configs/semi_markovian/semi_markovian.toml",
    #"configs/semi_markovian/semi_markovian_oracles.toml",
]

def main():
    try:
        for config_path in configs:
            cfg = load_config(config_path)
            run(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()