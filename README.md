# Dynamic Pricing and Networked Microgrid Trading Project

## Overview

This repository contains a full, end‑to‑end implementation of the experiments described in the paper **“Service Pricing and Networked Microgrid Trading for Heterogeneous Demand: A Causal Diffusion and Multi‑Agent Reinforcement Learning Approach.”**  The goal of the project is to provide a reproducible research framework that integrates causal modelling, non‑parametric clustering and multi‑agent reinforcement learning for two application scenarios:

1. **Dynamic service pricing** – multiple service providers adjust prices in real time for a population of customers with heterogeneous and time‑varying demand.  A conditional diffusion model learns the distribution of customer demand under different prices, a Dirichlet process clusters customers into latent groups and a multi‑agent actor‑critic algorithm optimises pricing strategies to maximise revenue while ensuring fairness and stability.
2. **Networked microgrid trading** – a group of microgrids with renewable generation and storage trade surplus energy among themselves and with the external grid.  The model combines local optimisation of battery dispatch, proportional bargaining for peer‑to‑peer trading, system marginal pricing for inter‑cluster trades and multi‑agent learning for bidding strategies.  The result is improved self‑sufficiency and reduced purchase/sale costs.

The repository is organised so that you can easily reproduce the experiments, extend the models, or adapt the framework to your own data.  All code is written in Python using widely‑used libraries such as **PyTorch**, **NumPy**, **Pandas** and **scikit‑learn**.

## Directory Structure

```
pricing_project/
├── README.md             – this file
├── requirements.txt      – list of Python dependencies
├── data/                 – sample data and scripts for generating synthetic data
│   ├── shopping_behavior.csv    – small sample of consumer shopping data (synthetic)
│   └── microgrid_data.csv       – synthetic microgrid time‑series data
├── src/                  – source code implementing models and training routines
│   ├── data_processing.py       – utilities for loading and preprocessing datasets
│   ├── diffusion_model.py       – conditional diffusion model for demand distribution
│   ├── dirichlet_clustering.py  – non‑parametric Bayesian clustering via Dirichlet process
│   ├── marl_pricing.py          – multi‑agent RL environment and actor‑critic algorithm for pricing
│   ├── microgrid_trading.py     – microgrid trading environment and RL bidding logic
│   ├── train_pricing.py         – script to train the pricing model end‑to‑end
│   └── train_microgrid.py       – script to train the microgrid trading model
└── .gitignore            – ignore Python bytecode and large files
```

## Installation

This project has been tested with **Python 3.9+**.  It is recommended to create a fresh virtual environment before installing the dependencies.

```bash
# clone the repository (replace with your own GitHub URL after upload)
git clone https://github.com/your‑username/pricing_project.git
cd pricing_project

# create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# install the required Python packages
pip install -r requirements.txt
```

The key dependencies include:

* **torch** for neural network models and reinforcement learning algorithms.
* **numpy** and **pandas** for numerical computations and data manipulation.
* **scikit‑learn** for clustering (Dirichlet process mixture models) and preprocessing utilities.
* **matplotlib** for plotting results (optional).

If you plan to use GPU acceleration, make sure to install the appropriate CUDA version of PyTorch.

## Data

Two synthetic datasets are included in the `data/` directory for demonstration and testing purposes:

* **`shopping_behavior.csv`** – a small, synthetic dataset emulating the Consumer Behavior & Shopping Habits data from Kaggle.  Each row represents a customer with five attributes (`age`, `gender`, `category`, `location`, and `past_purchases`), and an additional column `baseline_spend` approximating the customer’s historical spending.  In real experiments you should replace this file with your own data or download the original Kaggle dataset (the code in `data_processing.py` expects the same column names).
* **`microgrid_data.csv`** – synthetic time‑series data representing the power consumption and photovoltaic generation of three microgrids (residential, office and industrial).  The columns `load`, `solar` and `battery_capacity` contain normalised daily profiles.  Again, you can replace this with real Open Power System Data or your own microgrid measurements.

If you have access to the original datasets used in the paper, place them in the `data/` directory and modify the loader functions in `data_processing.py` accordingly.  The provided scripts are written to be flexible about the data source.

## Usage

### 1. Dynamic Pricing Experiment

To reproduce the dynamic pricing experiment:

```bash
# from the root of the repository
python src/train_pricing.py --epochs 100 --batch_size 64 --num_agents 3
```

This command will:

1. Load or generate customer attributes from `data/shopping_behavior.csv`.
2. Simulate baseline demand according to a mixed functional form as described in the paper.
3. Train a conditional diffusion model to learn the distribution of customer demand conditioned on price and customer features.
4. Cluster customers using a Dirichlet process mixture model.
5. Train a multi‑agent actor–critic model where each agent corresponds to a service provider and learns to set prices.
6. Output training logs, including rewards, fairness and price volatility metrics, and save model checkpoints.

You can customise the hyperparameters via command line arguments; see `python src/train_pricing.py --help` for details.

### 2. Microgrid Trading Experiment

To reproduce the microgrid trading experiment:

```bash
python src/train_microgrid.py --episodes 1000 --clusters 3
```

This script will:

1. Load or generate synthetic microgrid data from `data/microgrid_data.csv`.
2. Define a microgrid trading environment with local optimisation (battery dispatch), peer‑to‑peer trading within clusters and system marginal pricing across clusters.
3. Instantiate a multi‑agent reinforcement learning algorithm to learn optimal bids for each microgrid.
4. Train the model for the specified number of episodes and report performance metrics such as surplus reduction, purchase/sale cost reduction and social welfare.

Again, use `--help` to see all available options.

## Expected Output

During training the scripts will periodically print or log the following statistics:

* **Average reward** – the mean revenue per provider or microgrid over the evaluation horizon.
* **Fairness (Jain index)** – how evenly rewards are distributed across agents.
* **Price volatility** – the standard deviation of price sequences.
* **Social welfare** – total utility including both providers and customers.
* **Regret** – the gap between the obtained reward and the best known reward.
* **Energy trading metrics** – surplus reduction, purchase cost reduction and sale cost reduction for microgrids.

These metrics allow you to compare different algorithms or hyperparameter settings.  At the end of training the scripts will save learned model parameters in the `checkpoints/` directory (created automatically).

## Troubleshooting

* **Slow training or high memory usage** – reduce the number of diffusion steps, the neural network sizes or the replay buffer capacity.  You can also use CPU‑only training by avoiding GPU‑specific PyTorch builds.
* **Clustering fails to converge** – try adjusting the concentration parameter of the Dirichlet process (see `--alpha` in `train_pricing.py`) or increase the number of clusters manually.
* **Rewards remain low** – ensure that the synthetic data are generated correctly and that the neural networks have sufficient capacity.  Try increasing the number of training epochs or tuning the learning rate.
* **Microgrid results unstable** – the trading environment is inherently noisy; run multiple seeds (`--seed`) and average the results.  You can also adjust the reward shaping coefficients in `microgrid_trading.py`.

## Extending the Framework

This codebase is designed to be modular.  To extend or modify the experiments:

* Replace the synthetic datasets with real data by modifying the file paths and preprocessing logic in `data_processing.py`.
* Experiment with different diffusion architectures (e.g. U‑Net) or adjust the noise schedule in `diffusion_model.py`.
* Use alternative clustering methods (e.g. Gaussian mixture models, K‑means) by swapping out `dirichlet_clustering.py`.
* Implement other multi‑agent RL algorithms such as Proximal Policy Optimisation (PPO) or Q‑mixers, using the existing environment interfaces.
* Extend the microgrid model to include real‑time energy prices, stochastic renewable output or additional constraints.

## License

This project is released under the MIT License.  See `LICENSE` for details.

## Acknowledgements

This implementation was inspired by the models and experiments described in the paper, but is not an exact replica.  It is provided for educational purposes and to facilitate reproducible research on causal diffusion and multi‑agent reinforcement learning in dynamic pricing and energy systems.