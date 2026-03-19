import tomllib
from pathlib import Path
from cmab.utils.plotting import plot_regrets
from benchmarking.run import run

def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    cfg_path = Path("configs/markovian1.toml")
    cfg = load_config(cfg_path)
    run(cfg)


if __name__ == "__main__":
    main()