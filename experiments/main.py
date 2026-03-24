import tomllib
from pathlib import Path
from benchmarking.run import run

def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    cfg_path = Path("configs/iv/iv.toml")
    cfg = load_config(cfg_path)
    run(cfg)


if __name__ == "__main__":
    main()