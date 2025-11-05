"""Multi‑agent reinforcement learning environment and trainer for dynamic pricing.

This module defines a simple environment that models the interaction
between multiple service providers (agents) and a population of customers.
Each agent sets a price for its service; customers generate demand based
on their features and the offered price; and the agent receives revenue
equal to (price – cost) times demand.  The environment provides a
summarised state representation comprising cluster proportions and
estimated mean demand.  A multi‑agent actor–critic algorithm is used
to learn pricing strategies that maximise revenue while balancing
fairness and price stability.

The implementation is intentionally minimal and does not attempt to
exactly replicate the experiments in the paper.  Rather, it serves as a
starting point for readers to explore causal diffusion and multi‑agent
reinforcement learning in a reproducible setting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .data_processing import compute_true_demand


class PricingEnv:
    """Environment for dynamic pricing with heterogeneous demand.

    Parameters
    ----------
    customer_features : np.ndarray
        Array of shape `(n_customers, n_features)` containing normalised
        customer attributes.
    cluster_labels : np.ndarray
        Integer labels assigning each customer to a cluster.  The number of
        distinct labels defines the number of clusters.
    costs : List[float]
        Per‑unit cost for each provider/agent.
    price_range : Tuple[float, float]
        Minimum and maximum allowable prices.
    """

    def __init__(
        self,
        customer_features: np.ndarray,
        cluster_labels: np.ndarray,
        costs: List[float],
        price_range: Tuple[float, float] = (0.1, 5.0),
    ) -> None:
        self.customer_features = customer_features
        self.cluster_labels = cluster_labels
        self.num_agents = len(costs)
        self.costs = np.array(costs, dtype=np.float32)
        self.price_low, self.price_high = price_range
        # Normalise cluster proportions
        self.cluster_ids, self.cluster_counts = np.unique(cluster_labels, return_counts=True)
        self.cluster_proportions = self.cluster_counts / float(len(cluster_labels))
        self.n_clusters = len(self.cluster_ids)
        # State placeholders
        self.prev_prices = np.zeros(self.num_agents, dtype=np.float32)
        # Precompute true demand (mean) for each provider and customer
        self.true_demand_matrix = np.stack(
            [compute_true_demand(self.customer_features, i) for i in range(self.num_agents)],
            axis=0,
        )  # shape (num_agents, n_customers)
        # Compute per‑agent mean demand across customers
        self.mean_true_demand = self.true_demand_matrix.mean(axis=1)  # shape (num_agents,)

    def reset(self) -> List[np.ndarray]:
        """Reset the environment to its initial state.

        Returns:
            A list of state vectors, one per agent.
        """
        self.prev_prices.fill(0.0)
        # Construct state for each agent: [mean_true_demand, cluster_proportions..., prev_price]
        states = []
        for i in range(self.num_agents):
            state = np.concatenate(
                [
                    np.array([self.mean_true_demand[i]], dtype=np.float32),
                    self.cluster_proportions.astype(np.float32),
                    np.array([self.prev_prices[i]], dtype=np.float32),
                ]
            )
            states.append(state)
        return states

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, bool, dict]:
        """Simulate a single pricing step.

        Args:
            actions: Array of shape `(num_agents,)` containing the prices chosen
                by each agent.

        Returns:
            next_states: List of state vectors for each agent.
            rewards: Array of rewards for each agent.
            done: Boolean flag indicating whether the episode should terminate.
            info: Dictionary with additional diagnostic information.
        """
        # Clamp prices to the allowed range
        prices = np.clip(actions.astype(np.float32), self.price_low, self.price_high)
        # Compute realised demand for each provider (mean across customers)
        demand = np.maximum(0.0, self.mean_true_demand - 0.1 * prices)
        # Revenue minus cost
        rewards = prices * demand - self.costs * demand
        # Update previous prices
        self.prev_prices = prices
        # Compute next state
        next_states = []
        for i in range(self.num_agents):
            s = np.concatenate(
                [
                    np.array([self.mean_true_demand[i]], dtype=np.float32),
                    self.cluster_proportions.astype(np.float32),
                    np.array([self.prev_prices[i]], dtype=np.float32),
                ]
            )
            next_states.append(s)
        done = False
        info = {}
        return next_states, rewards, done, info


class ActorCriticNetwork(nn.Module):
    """Combined actor and critic network for a single agent.

    The actor outputs the mean and log standard deviation of a Gaussian
    distribution over continuous actions (prices).  The critic outputs an
    estimate of the state value.  Both share the first hidden layer but
    then branch into separate heads.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Linear(hidden_dim, 1)
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(state))
        actor_h = F.relu(self.actor_fc(h))
        critic_h = F.relu(self.critic_fc(h))
        mean = self.actor_mean(actor_h)
        log_std = self.actor_log_std(actor_h)
        value = self.critic_out(critic_h)
        return mean.squeeze(-1), log_std.squeeze(-1), value.squeeze(-1)


class Agent:
    """Actor‑critic agent for continuous action spaces."""

    def __init__(self, state_dim: int, lr: float = 1e-3, gamma: float = 0.95) -> None:
        self.gamma = gamma
        self.network = ActorCriticNetwork(state_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = []  # stores (state, action, log_prob, reward, done, value)

    def select_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Select an action according to the current policy.

        Args:
            state: State vector as a numpy array.

        Returns:
            action: Sampled price (float).
            log_prob: Log probability of the sampled action.
            value: Critic value estimate for the state.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        mean, log_std, value = self.network(state_t)
        std = log_std.exp()
        # Sample from a Gaussian distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.buffer.append((state, action, log_prob, reward, value, done))

    def finish_episode(self) -> None:
        """Compute returns/advantages and update network parameters."""
        # Compute returns and advantages
        returns = []
        advantages = []
        G = 0
        for _, _, _, reward, value, done in reversed(self.buffer):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # Compute advantages (returns minus value)
        values = torch.tensor([b[4] for b in self.buffer], dtype=torch.float32)
        advantages = returns - values
        states = torch.stack([torch.from_numpy(b[0]).float() for b in self.buffer])
        actions = torch.tensor([b[1] for b in self.buffer], dtype=torch.float32)
        log_probs_old = torch.tensor([b[2] for b in self.buffer], dtype=torch.float32)
        # Recompute log_probs and values for gradient calculation
        mean, log_std, values_new = self.network(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        # Loss components
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = F.mse_loss(values_new, returns)
        # Entropy regularisation to encourage exploration
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Clear buffer
        self.buffer = []


class MultiAgentTrainer:
    """Coordinator for training multiple agents in the pricing environment."""

    def __init__(
        self,
        env: PricingEnv,
        num_agents: int,
        gamma: float = 0.95,
        lr: float = 1e-3,
    ) -> None:
        self.env = env
        self.agents: List[Agent] = []
        self.num_agents = num_agents
        # Determine state dimension: mean_true_demand (1) + cluster proportions (K) + prev_price (1)
        state_dim = 1 + env.n_clusters + 1
        for _ in range(num_agents):
            self.agents.append(Agent(state_dim, lr=lr, gamma=gamma))

    def train(
        self,
        episodes: int,
        steps_per_episode: int = 10,
        verbose: bool = True,
    ) -> None:
        """Train the agents for a number of episodes.

        Args:
            episodes: Total number of episodes.
            steps_per_episode: Length of each episode (in steps).
            verbose: If True, print progress information.
        """
        for ep in range(episodes):
            states = self.env.reset()
            ep_rewards = np.zeros(self.num_agents, dtype=np.float32)
            for step in range(steps_per_episode):
                actions = np.zeros(self.num_agents, dtype=np.float32)
                log_probs = []
                values = []
                for i, agent in enumerate(self.agents):
                    action, log_prob, value = agent.select_action(states[i])
                    actions[i] = action
                    log_probs.append(log_prob)
                    values.append(value)
                next_states, rewards, done, _ = self.env.step(actions)
                # Store transitions
                for i, agent in enumerate(self.agents):
                    agent.store_transition(states[i], actions[i], log_probs[i], rewards[i], values[i], done)
                    ep_rewards[i] += rewards[i]
                states = next_states
                if done:
                    break
            # Finish episode: update policies
            for agent in self.agents:
                agent.finish_episode()
            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                avg_reward = ep_rewards.mean()
                print(f"Episode {ep+1}/{episodes}, mean reward per agent: {avg_reward:.3f}")