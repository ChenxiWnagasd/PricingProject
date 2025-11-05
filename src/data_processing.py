"""Utilities for loading and generating datasets.

This module defines helper functions to load the synthetic datasets included with
the repository and to generate synthetic data on the fly.  It also provides
functions to compute ground‑truth demand for the dynamic pricing experiment.

The dynamic pricing scenario assumes that each customer is described by a
five‑dimensional feature vector `(age, gender, category, location, past_purchases)`
and a baseline spending amount.  A mixed non‑linear function maps these
features to demand for each provider.  This function is used both to simulate
observed demand and as the ground‑truth target when training the diffusion
model.

The microgrid scenario assumes time‑series measurements of `load`, `solar` and
`battery_capacity` for each microgrid.  These values can be loaded from
`data/microgrid_data.csv` or generated synthetically.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

def load_shopping_data(path: str) -> pd.DataFrame:
    """Load shopping/customer data from a CSV file.

    The CSV file should contain the columns:

    * age
    * gender (0 or 1)
    * category (categorical integer)
    * location (categorical integer)
    * past_purchases (integer count)
    * baseline_spend (float)

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame with the expected columns.  If the file cannot be
        loaded or is missing required columns, an exception is raised.
    """
    df = pd.read_csv(path)
    required_cols = [
        "age",
        "gender",
        "category",
        "location",
        "past_purchases",
        "baseline_spend",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[required_cols].copy()


def generate_synthetic_shopping_data(n_samples: int = 600, random_state: int | None = None) -> pd.DataFrame:
    """Generate a synthetic shopping dataset.

    This function draws random values for five customer attributes from
    reasonable ranges and computes a baseline spending amount.  The ranges
    loosely approximate those in the Kaggle shopping behaviour dataset.

    Args:
        n_samples: Number of customers to generate.
        random_state: Optional random seed for reproducibility.

    Returns:
        A DataFrame with columns `[age, gender, category, location, past_purchases, baseline_spend]`.
    """
    rng = np.random.default_rng(random_state)
    # Ages between 18 and 65
    ages = rng.integers(18, 66, size=n_samples)
    # Gender 0/1
    genders = rng.integers(0, 2, size=n_samples)
    # Categorical product category (0–4)
    categories = rng.integers(0, 5, size=n_samples)
    # Location categories (0–3)
    locations = rng.integers(0, 4, size=n_samples)
    # Past purchases (0–10)
    past = rng.integers(0, 11, size=n_samples)
    # Baseline spend is drawn from a log‑normal distribution scaled by category and past purchases
    baseline = np.exp(rng.normal(loc=4.5, scale=0.5, size=n_samples))
    baseline *= (1 + 0.1 * categories) * (1 + 0.05 * past)
    df = pd.DataFrame(
        {
            "age": ages,
            "gender": genders,
            "category": categories,
            "location": locations,
            "past_purchases": past,
            "baseline_spend": baseline,
        }
    )
    return df


def compute_true_demand(features: np.ndarray, provider_idx: int) -> np.ndarray:
    """Compute the ground‑truth demand for a given provider using a mixed function.

    The function combines linear, interaction and non‑linear terms of the
    customer attributes to produce a synthetic demand value.  Each provider
    has slightly different coefficients.

    Args:
        features: Array of shape `(n_customers, n_features)` containing
            `[age, gender, category, location, past_purchases]` (normalised).
        provider_idx: Index of the provider (0, 1, 2, ...).

    Returns:
        A 1‑D numpy array of length `n_customers` with the true demand values.
    """
    x = features.astype(float)
    # Normalise age into [0,1]
    age = x[:, 0] / 65.0
    gender = x[:, 1]
    category = x[:, 2] / 4.0
    location = x[:, 3] / 3.0
    past = x[:, 4] / 10.0
    # Provider‑specific weights
    rng = np.random.default_rng(42 + provider_idx)
    w = rng.uniform(-1.0, 1.0, size=5)
    # Linear component
    linear = w[0] * age + w[1] * gender + w[2] * category + w[3] * location + w[4] * past
    # Interaction terms
    interaction = (
        0.5 * age * category + 0.3 * gender * past + 0.2 * location * category
    )
    # Non‑linear (sinusoidal) term
    nonlinear = 0.4 * np.sin(2 * math.pi * past) + 0.6 * np.sin(2 * math.pi * age)
    demand = np.maximum(0.0, 1.0 + linear + interaction + nonlinear)
    return demand


def load_microgrid_data(path: str) -> pd.DataFrame:
    """Load microgrid time‑series data from a CSV file.

    The CSV is expected to contain columns:

    * microgrid_id – integer identifier for each microgrid
    * time – time index (e.g. hour of day)
    * load – normalised load consumption
    * solar – normalised solar generation
    * battery_capacity – capacity of the local battery

    Args:
        path: Path to the microgrid CSV file.

    Returns:
        A pandas DataFrame.
    """
    df = pd.read_csv(path)
    required = {"microgrid_id", "time", "load", "solar", "battery_capacity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df.copy()


def generate_synthetic_microgrid_data(
    n_microgrids: int = 3,
    n_steps: int = 24,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic microgrid data for demonstration.

    Each microgrid is assigned a random daily load and solar generation profile.
    The battery capacity is fixed at 1.0 for all microgrids.  Values are
    normalised to be between 0 and 1.

    Args:
        n_microgrids: Number of microgrids to simulate.
        n_steps: Number of time steps (e.g. 24 for one day of hourly data).
        random_state: Optional seed.

    Returns:
        A DataFrame with columns `[microgrid_id, time, load, solar, battery_capacity]`.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for mg in range(n_microgrids):
        # Generate a daily load curve: base + morning/evening peaks
        t = np.linspace(0, 23, n_steps)
        base_load = 0.4 + 0.3 * np.sin(2 * math.pi * (t - 7) / 24)
        # Add random noise
        noise = rng.normal(scale=0.05, size=n_steps)
        load = np.clip(base_load + noise, 0.2, 1.2)
        # Solar generation: only during daylight hours
        solar_profile = np.clip(0.8 * np.sin(math.pi * (t - 6) / 12), 0.0, 1.0)
        solar = solar_profile + rng.normal(scale=0.05, size=n_steps)
        solar = np.clip(solar, 0.0, 1.0)
        battery_capacity = np.ones(n_steps)
        for ti in range(n_steps):
            rows.append(
                {
                    "microgrid_id": mg,
                    "time": ti,
                    "load": float(load[ti]),
                    "solar": float(solar[ti]),
                    "battery_capacity": float(battery_capacity[ti]),
                }
            )
    return pd.DataFrame(rows)