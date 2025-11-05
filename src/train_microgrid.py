#!/usr/bin/env python
"""Entry point for training the microgrid trading experiment.

This script loads synthetic or real microgrid time‑series data and
trains a multi‑agent reinforcement learning model to learn bidding
strategies.  The environment implements a simplified energy trading
mechanism where microgrids buy or sell energy depending on their
net demand and bids.  The RL algorithm is an actor–critic method with
independent agents.

Example usage:

```bash
python src/train_microgrid.py --episodes 200
```
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_processing import load_microgrid_data, generate_synthetic_microgrid_data
from microgrid_trading import MicrogridEnv, MultiAgentMGTrainer


def main(args: argparse.Namespace) -> None:
    # Load or generate microgrid data
    if args.use_synthetic:
        df = generate_synthetic_microgrid_data(
            n_microgrids=args.num_microgrids, n_steps=args.steps, random_state=args.seed
        )
    else:
        data_path = args.data_path or os.path.join(os.path.dirname(__file__), "..", "data", "microgrid_data.csv")
        df = load_microgrid_data(data_path)
        # Optionally filter by number of microgrids and steps
        df = df[df["microgrid_id"] < args.num_microgrids]
        df = df[df["time"] < args.steps]
    # Convert to numpy array
    data_array = df[["microgrid_id", "time", "load", "solar", "battery_capacity"]].values.astype(np.float32)
    # Instantiate environment
    env = MicrogridEnv(
        data=data_array,
        grid_price=args.grid_price,
        price_range=(args.price_low, args.price_high),
    )
    # Instantiate trainer
    trainer = MultiAgentMGTrainer(env=env, lr=args.learning_rate, gamma=args.gamma)
    # Train
    print("Training microgrid bidding agents...")
    trainer.train(episodes=args.episodes, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train microgrid trading agents")
    parser.add_argument(
        "--use_synthetic", action="store_true", help="Generate synthetic microgrid data"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to microgrid CSV (ignored if synthetic)"
    )
    parser.add_argument(
        "--num_microgrids", type=int, default=3, help="Number of microgrids to include"
    )
    parser.add_argument(
        "--steps", type=int, default=24, help="Number of time steps per episode (e.g. hours)"
    )
    parser.add_argument(
        "--grid_price", type=float, default=1.0, help="Exogenous grid price"
    )
    parser.add_argument(
        "--price_low", type=float, default=0.0, help="Minimum bid price"
    )
    parser.add_argument(
        "--price_high", type=float, default=2.0, help="Maximum bid price"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for agents"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for RL"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()
    main(args)