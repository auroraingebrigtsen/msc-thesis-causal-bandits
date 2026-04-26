import tomllib
from pathlib import Path
from benchmarking.run import run

def load_config(path: str):
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    try:
        cfg_path = Path("configs/markovian/large_changes.toml")
        cfg = load_config(cfg_path)
        run(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()