"""Simple conditional diffusion model for demand distribution estimation.

This module implements a lightweight version of the denoising diffusion
probabilistic model (DDPM) specialised for one‑dimensional conditional data.
The model learns to map Gaussian noise into samples drawn from the
distribution of customer demand given price and customer features.  It
supports training via the standard mean squared error loss on the
predicted noise and sampling via the reverse diffusion process.

The implementation is intentionally simple and avoids dependencies on
external deep learning frameworks beyond PyTorch.  It uses a multilayer
perceptron (MLP) to approximate the score function.  For a full‑fledged
implementation you may consider integrating more expressive neural
architectures such as U‑Nets or attention mechanisms.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_time_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal position/time embeddings.

    Maps a batch of discrete time steps into a continuous embedding using
    sinusoidal functions.  This technique was popularised in the transformer
    literature and is used here to embed diffusion timesteps.

    Args:
        timesteps: A 1‑D tensor of integer time steps.
        embedding_dim: Dimension of the embedding.

    Returns:
        A tensor of shape `(len(timesteps), embedding_dim)`.
    """
    # Expand to float and compute the sinusoidal frequencies
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
    )
    freqs = freqs.to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 == 1:
        # Pad an extra zero if embedding_dim is odd
        embedding = F.pad(embedding, (0, 1))
    return embedding


class NoisePredictor(nn.Module):
    """Simple MLP that predicts the noise in a diffusion step.

    The network takes as input the noised data `x_t`, a conditioning vector
    `cond` and a time embedding `t_emb` and outputs the predicted noise
    epsilon.  The architecture concatenates these inputs and passes them
    through fully connected layers with ReLU activations.
    """

    def __init__(self, x_dim: int, cond_dim: int, time_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(x_dim + cond_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_t, cond, t_emb], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class ConditionalDiffusionModel:
    """Encapsulates the forward and reverse diffusion processes.

    The class manages the scheduling of noise variances, the training
    loop and sampling.  It does not implement the orthogonal loss from
    the paper; instead, it uses the standard MSE loss on the predicted
    noise.  Conditioning information (prices and customer features) is
    concatenated into a single vector and fed into the MLP at each step.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        timesteps: int = 200,
        hidden_dim: int = 128,
        device: str | torch.device = "cpu",
    ) -> None:
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.T = timesteps
        self.device = torch.device(device)
        # Linear beta schedule
        beta = torch.linspace(1e-4, 0.02, self.T)
        self.beta = beta.to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.noise_predictor = NoisePredictor(
            x_dim=data_dim, cond_dim=cond_dim, time_dim=64, hidden_dim=hidden_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.noise_predictor.parameters(), lr=1e-3)

    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: sample q(x_t | x_0)."""
        alphas_cumprod_t = self.alpha_cumprod[t].view(-1, 1).to(self.device)
        sqrt_alpha = torch.sqrt(alphas_cumprod_t)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_cumprod_t)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def train_step(self, x0: torch.Tensor, cond: torch.Tensor) -> float:
        """Perform a single training step on a batch of data.

        Args:
            x0: Clean data samples, shape (batch_size, data_dim).
            cond: Conditioning vectors, shape (batch_size, cond_dim).

        Returns:
            The training loss as a float.
        """
        bsz = x0.shape[0]
        # Sample random timesteps for each sample
        t = torch.randint(0, self.T, (bsz,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self._q_sample(x0, t, noise)
        # Compute time embeddings
        t_emb = _build_time_embedding(t, 64)
        # Predict noise
        pred_noise = self.noise_predictor(x_t, cond, t_emb)
        loss = F.mse_loss(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(
        self,
        data_x0: np.ndarray,
        cond: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        print_every: int = 10,
    ) -> None:
        """Train the diffusion model on the provided data.

        Args:
            data_x0: Array of clean demand samples (shape `(n_samples, data_dim)`).
            cond: Array of conditioning vectors (shape `(n_samples, cond_dim)`).
            epochs: Number of passes over the dataset.
            batch_size: Mini‑batch size.
            print_every: Interval for printing loss information.
        """
        dataset_size = data_x0.shape[0]
        indices = np.arange(dataset_size)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]
                x0_batch = torch.from_numpy(data_x0[idx]).float().to(self.device)
                cond_batch = torch.from_numpy(cond[idx]).float().to(self.device)
                loss = self.train_step(x0_batch, cond_batch)
                total_loss += loss
                n_batches += 1
            if (epoch + 1) % print_every == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    @torch.no_grad()
    def sample(self, cond: np.ndarray, num_steps: int | None = None) -> np.ndarray:
        """Generate samples from the learned distribution given conditioning.

        Args:
            cond: Conditioning vectors, shape `(n_samples, cond_dim)`.
            num_steps: Number of reverse diffusion steps.  If `None`, use the
                full schedule length `self.T`.

        Returns:
            A numpy array of generated samples with shape `(n_samples, data_dim)`.
        """
        if num_steps is None:
            num_steps = self.T
        n = cond.shape[0]
        device = self.device
        x_t = torch.randn(n, self.data_dim, device=device)
        cond_t = torch.from_numpy(cond).float().to(device)
        for t in reversed(range(num_steps)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            t_emb = _build_time_embedding(t_batch, 64)
            # Predict noise
            eps = self.noise_predictor(x_t, cond_t, t_emb)
            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            # Compute the mean of the posterior
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * eps)
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta_t)
                x_t = mean + sigma * noise
            else:
                x_t = mean
        return x_t.cpu().numpy()