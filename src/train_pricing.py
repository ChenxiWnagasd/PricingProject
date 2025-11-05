#!/usr/bin/env python
"""Entry point to train the dynamic pricing experiment.

This script loads (or generates) customer data, fits a Dirichlet process
mixture model to cluster customers, trains a conditional diffusion model
to estimate demand distributions and finally trains a multi‑agent
reinforcement learning model to learn pricing strategies.  The
implementation uses simplified models and should be viewed as a
demonstrative baseline rather than a production‑ready system.

Usage example:

```bash
python src/train_pricing.py --epochs 50 --num_agents 3 --use_synthetic
```
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from data_processing import (
    load_shopping_data,
    generate_synthetic_shopping_data,
    compute_true_demand,
)
from dirichlet_clustering import DirichletProcessClustering
from diffusion_model import ConditionalDiffusionModel
from marl_pricing import PricingEnv, MultiAgentTrainer


def main(args: argparse.Namespace) -> None:
    # Load or generate customer data
    if args.use_synthetic:
        df = generate_synthetic_shopping_data(n_samples=args.num_customers, random_state=args.seed)
    else:
        # Expect file to exist in data/ directory
        data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "data", "shopping_behavior.csv")
        df = load_shopping_data(data_path)
        if args.num_customers is not None and args.num_customers < len(df):
            df = df.sample(n=args.num_customers, random_state=args.seed).reset_index(drop=True)
    # Extract features and normalize to [0,1]
    features = df[["age", "gender", "category", "location", "past_purchases"]].values.astype(np.float32)
    # Scale continuous features
    features[:, 0] /= 65.0  # age
    features[:, 1] = features[:, 1]  # gender is 0/1
    features[:, 2] /= max(1.0, features[:, 2].max())
    features[:, 3] /= max(1.0, features[:, 3].max())
    features[:, 4] /= max(1.0, features[:, 4].max())
    # Dirichlet process clustering
    clustering = DirichletProcessClustering(max_components=args.max_clusters, random_state=args.seed)
    cluster_labels = clustering.fit_predict(features)
    num_clusters = clustering.get_num_clusters()
    print(f"Identified {num_clusters} customer clusters")
    # Prepare conditional diffusion training data
    # Randomly generate price samples for each customer and provider
    num_agents = args.num_agents
    n_samples = len(df) * num_agents
    conds = []
    demands = []
    # Generate synthetic training pairs: (price, features) -> true demand
    rng = np.random.default_rng(args.seed)
    for i in range(num_agents):
        # Simulate random prices in [0.1, 5.0]
        prices = rng.uniform(0.1, 5.0, size=len(df))
        demand_true = []
        for j in range(len(df)):
            # For simplicity we use the true demand without price impact plus noise
            base_demand = 1.0 + compute_true_demand(features[j:j+1] * 1.0, provider_idx=i)[0]
            noise = rng.normal(scale=0.05)
            d = max(0.0, base_demand - 0.1 * prices[j] + noise)
            demand_true.append(d)
        # Conditioning vector: [price] + features
        cond_i = np.hstack([prices.reshape(-1, 1), features])
        conds.append(cond_i)
        demands.append(np.array(demand_true).reshape(-1, 1))
    conds_arr = np.vstack(conds).astype(np.float32)
    demands_arr = np.vstack(demands).astype(np.float32)
    # Train diffusion model
    diff_model = ConditionalDiffusionModel(
        data_dim=1,
        cond_dim=1 + features.shape[1],
        timesteps=args.diffusion_steps,
        hidden_dim=args.diffusion_hidden,
        device=args.device,
    )
    print("Training diffusion model...")
    diff_model.train_model(
        data_x0=demands_arr,
        cond=conds_arr,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
        print_every=max(1, args.diffusion_epochs // 5),
    )
    # Set up pricing environment
    # Use mean true demand for each provider as estimated by the diffusion model (for simplicity)
    # Compute costs for providers (slightly different)
    costs = [0.3 + 0.05 * i for i in range(num_agents)]
    env = PricingEnv(
        customer_features=features,
        cluster_labels=cluster_labels,
        costs=costs,
        price_range=(0.1, 5.0),
    )
    # Train multi‑agent RL
    trainer = MultiAgentTrainer(
        env=env,
        num_agents=num_agents,
        gamma=args.gamma,
        lr=args.learning_rate,
    )
    print("Training multi‑agent pricing model...")
    trainer.train(
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        verbose=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dynamic pricing experiment")
    parser.add_argument(
        "--num_agents", type=int, default=3, help="Number of service providers (agents)"
    )
    parser.add_argument(
        "--num_customers",
        type=int,
        default=600,
        help="Number of customers to use (only for synthetic data or to subsample real data)",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=5,
        help="Maximum number of clusters for Dirichlet process mixture",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Generate synthetic shopping data instead of loading from file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the shopping data CSV (ignored when using synthetic data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--diffusion_steps", type=int, default=200, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--diffusion_hidden", type=int, default=128, help="Hidden dimension in diffusion model"
    )
    parser.add_argument(
        "--diffusion_epochs", type=int, default=5, help="Epochs to train the diffusion model"
    )
    parser.add_argument(
        "--diffusion_batch_size", type=int, default=64, help="Batch size for diffusion model"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for training (cpu or cuda)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of RL training episodes"
    )
    parser.add_argument(
        "--steps_per_episode", type=int, default=10, help="Number of steps per episode"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="Discount factor for RL"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for RL agents"
    )
    args = parser.parse_args()
    main(args)