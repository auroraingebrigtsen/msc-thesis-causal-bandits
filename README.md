# msc-thesis-causal-bandits
Code and experiments for my master thesis (INF399 @uib) on non-stationary causal bandits.


## Modules

The repository contains two modules under `src`.

### Benchmarking Module
Provides the infrastructure for running `.toml`-based experiments.  
It includes:

- reusable bandit environments
- an agent factory for constructing algorithms
- a `run` entry point for executing experiments

The `run` function evaluates a set of bandit algorithms in a specified environment for **T** time steps and reports results averaged over **n** runs using the provided random seed.

### CMAB Module
Contains the causal bandit implementation.  
This module includes:

- a general causal bandit environment based on a structural causal model (SCM)
- support for non-stationary schedules
- implementations of different bandit algorithms

## Usage
Make sure you have uv installed

### 1. Create a virtual environment
```bash
uv venv
```

### 2. Activate the venv
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Configure an experiment
- Create a ```.toml``` file in the ```experiments/configs``` folder
- Use the ```experiments/configs/example_config.toml``` to specify the values of the agents and environment (including the SCM and the change schedule)
- Point the entry point at your config file in  ```experiments/main.py```
- Run the experiment by entering ```experiments``` and running the command:
 ```bash
uv run main.py
```

### 5. Create a new environment
There already exist several causal bandit environments in the ```benchmarking``` module. If you want to create a new one, follow the steps below:
- Create a file in the ```src/benchmarking/environments``` folder
- Specify the components of the SCM in the file
- Include it in the builder function in the ```src/benchmarking/environments/__init__.py``` file

### 6. Create a new bandit algorithm
The ```cmab``` module contains several bandit algorithms. If you want to create a new one, follow the steps below:
- Create a file in the ```src/cmab/algorithms``` folder (or place it in the folder where it belongs)
- Implement an instance of the ```BaseBanditAlgorithm``` 
- Register it in ```src/cmab/algorithms/__init__.py``` file

