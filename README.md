# Dynamic Blockchain Optimization Framework 

This repository contains the official Python simulation codebase for the journal paper;
  The  Dynamic Framework For Blockchain Optimization: Adaptive Trilemma Balancing (ATB), Quadrilemma, And Meta-Trilemma.
The suite is designed to model, optimize, and simulate dynamic blockchain operations, balancing critical performance trade-offs.

File Structure & Explanations
The repository is divided into multiple modular Python scripts to support the advance reinforcement learning and game-theoretic simulations. 

* `__init__.py`: Initializes the simulation module, allowing the codebase to be imported cleanly.
* `blockchain_env.py`: Defines the custom simulation environment, modeling block generation, network latency, and consensus mechanisms.
* `data_bridge.py`: Acts as the translation layer, converting blockchain telemetry and simulation data into formats compatible with the optimization and RL models.
* `game_theory.py`: Models the strategic interactions, utility functions, and payoff matrices of network validators to predict network equilibrium.
* `main.py`: The core entry point to execute the full simulation pipeline. This script is used to run  defined experiments and benchmark tests.
* `metrics.py`: Contains evaluation functions and statistical utilities to compute efficiency, decentralization, security, and scalability metrics.
* `mpc_controller.py`: Implements the Model Predictive Control logic, forecasting future states and dynamically adjusting parameters to stabilize network performance.
* `pareto_optimizer.py`: Handles multi-objective optimization, ensuring that the adjustments made respect Pareto efficiency boundaries.
* `plots.py`: A visualization utility for rendering performance graphs, charting trilemma balancing, and generating plots for the journal paper.
* `rl_agent.py`: The custom Reinforcement Learning agent implementation designed to learn optimal network configuration policies.
* `rl_agent-sb3.py`: Wraps the RL models using the Stable-Baselines3 library to leverage off-the-shelf RL algorithms.
* `rl_train.py`: The training script dedicated to training the custom `rl_agent.py` models over simulated time horizons.
* `rl_train-sb3.py`: The training pipeline specifically tailored for training the Stable-Baselines3 agents.
* `setup.py`: The installation script. This is used to configure dependencies and package the project.

# Getting Started

# Prerequisites
* Python 3.9+ 
* pip (Python package installer)
#Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd A-Dynamic-Framework-For-Blockchain-Optimization
   ```
2. **Install dependencies:**
   ```bash
   pip install -e .
   ```
# Running the Simulation
To execute the full dynamic optimization and training sequence, run:
```bash
python main.py
```

# License
This project is licensed under the MIT LICENSE.
