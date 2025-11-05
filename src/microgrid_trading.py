"""Multi‑agent environment and training for networked microgrid trading.

This simplified module implements a toy version of the microgrid energy
trading problem.  A set of microgrids each have a time‑varying net
energy balance (load minus solar generation).  At each time step each
microgrid places a bid indicating the price at which it is willing to
buy or sell energy.  The environment matches bids through a simple
clearing mechanism and computes rewards based on the traded energy and
prices.  The goal of each agent is to maximise revenue (or minimise
cost) over the course of an episode.

Note: This implementation omits battery dynamics, local optimisation and
system marginal pricing mechanisms described in the paper.  It is
provided as a starting point for experimentation and can be extended
with more realistic models (e.g. proportional bargaining, battery
constraints, peer‑to‑peer trading networks, etc.).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .data_processing import load_microgrid_data


class MicrogridEnv:
    """Environment representing simplified energy trading among microgrids.

    Each microgrid has a time‑series of load and solar generation.  The net
    energy `net = load - solar` is positive when the microgrid needs to
    purchase energy and negative when it has surplus energy to sell.  At
    each time step each microgrid bids a price; the environment clears
    trades by pairing buyers and sellers at an average clearing price.

    Parameters
    ----------
    data : np.ndarray
        Array with columns `(microgrid_id, time, load, solar, battery_capacity)`.
    grid_price : float
        Exogenous price for buying/selling energy from/to the main grid.
    price_range : Tuple[float, float]
        Minimum and maximum bid values.
    """

    def __init__(
        self,
        data: np.ndarray,
        grid_price: float = 1.0,
        price_range: Tuple[float, float] = (0.0, 2.0),
    ) -> None:
        # Data sorted by microgrid_id then time
        self.data = data
        self.grid_price = grid_price
        self.price_low, self.price_high = price_range
        # Determine number of microgrids and time horizon
        self.microgrid_ids = np.unique(data[:, 0].astype(int))
        self.num_agents = len(self.microgrid_ids)
        self.times = np.unique(data[:, 1].astype(int))
        self.n_steps = len(self.times)
        # Precompute net demand per microgrid and time
        self.net_demand = {}  # dict: (mg_id, time) -> net
        for row in data:
            mg, t, load, solar, _ = row
            self.net_demand[(int(mg), int(t))] = load - solar
        # State trackers
        self.current_step = 0
        self.prev_bids = np.zeros(self.num_agents, dtype=np.float32)

    def reset(self) -> List[np.ndarray]:
        """Reset the environment to the beginning of the episode."""
        self.current_step = 0
        self.prev_bids.fill(0.0)
        return self._get_states()

    def _get_states(self) -> List[np.ndarray]:
        """Construct state representation for each microgrid."""
        states: List[np.ndarray] = []
        t_norm = self.current_step / max(1, self.n_steps - 1)
        for idx, mg_id in enumerate(self.microgrid_ids):
            net = self.net_demand[(mg_id, self.current_step)]  # positive = need to buy
            # State vector: [net, time_norm, prev_bid]
            states.append(
                np.array([net, t_norm, self.prev_bids[idx]], dtype=np.float32)
            )
        return states

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, bool, dict]:
        """Perform a single trading step.

        Args:
            actions: Array of bids (shape `(num_agents,)`).

        Returns:
            next_states, rewards, done, info
        """
        bids = np.clip(actions.astype(np.float32), self.price_low, self.price_high)
        # Compute net energy for each microgrid at this step
        nets = np.array(
            [self.net_demand[(int(mg), self.current_step)] for mg in self.microgrid_ids],
            dtype=np.float32,
        )
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        # Determine total surplus and deficit
        surplus_mask = nets < 0  # sellers
        deficit_mask = nets > 0  # buyers
        total_surplus = -np.sum(nets[surplus_mask]) if np.any(surplus_mask) else 0.0
        total_deficit = np.sum(nets[deficit_mask]) if np.any(deficit_mask) else 0.0
        # Clearing price: average of bids from active traders or grid price if none
        if np.any(surplus_mask) and np.any(deficit_mask):
            clearing_price = (np.mean(bids[surplus_mask]) + np.mean(bids[deficit_mask])) / 2.0
        else:
            clearing_price = self.grid_price
        # For each microgrid compute reward
        for idx in range(self.num_agents):
            net = nets[idx]
            bid = bids[idx]
            if net < 0:  # surplus: sell energy
                energy = -net
                # Sell to buyers first; if buyers insufficient then sell to grid
                traded_energy = energy if total_deficit >= energy else total_deficit
                revenue = traded_energy * clearing_price + (energy - traded_energy) * self.grid_price
                rewards[idx] = revenue
            elif net > 0:  # deficit: buy energy
                energy = net
                # Buy from sellers first
                bought_energy = energy if total_surplus >= energy else total_surplus
                cost = bought_energy * clearing_price + (energy - bought_energy) * self.grid_price
                rewards[idx] = -cost
            else:
                rewards[idx] = 0.0
        # Update step counter and previous bids
        self.prev_bids = bids
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_states = self._get_states() if not done else [np.zeros_like(s) for s in self._get_states()]
        info = {"clearing_price": clearing_price}
        return next_states, rewards, done, info


class ActorCriticMG(nn.Module):
    """Combined actor and critic network for microgrid bidding."""

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


class MGAgent:
    """Actor‑critic agent for microgrid bidding."""

    def __init__(self, state_dim: int, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.gamma = gamma
        self.net = ActorCriticMG(state_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = []

    def select_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        mean, log_std, value = self.net(state_t)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.buffer.append((state, action, log_prob, reward, value, done))

    def finish_episode(self) -> None:
        # Compute returns
        returns = []
        G = 0.0
        for _, _, _, reward, _, done in reversed(self.buffer):
            if done:
                G = 0.0
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        states = torch.stack([torch.from_numpy(b[0]).float() for b in self.buffer])
        actions = torch.tensor([b[1] for b in self.buffer], dtype=torch.float32)
        old_log_probs = torch.tensor([b[2] for b in self.buffer], dtype=torch.float32)
        values_old = torch.tensor([b[4] for b in self.buffer], dtype=torch.float32)
        advantages = returns - values_old
        mean, log_std, values = self.net(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.buffer = []


class MultiAgentMGTrainer:
    """Trainer coordinating multiple microgrid agents."""

    def __init__(self, env: MicrogridEnv, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.env = env
        self.agents: List[MGAgent] = []
        state_dim = 3  # [net, time_norm, prev_bid]
        for _ in range(env.num_agents):
            self.agents.append(MGAgent(state_dim, lr=lr, gamma=gamma))

    def train(self, episodes: int, verbose: bool = True) -> None:
        for ep in range(episodes):
            states = self.env.reset()
            ep_reward = np.zeros(self.env.num_agents, dtype=np.float32)
            done = False
            while not done:
                actions = np.zeros(self.env.num_agents, dtype=np.float32)
                log_probs = []
                values = []
                for i, agent in enumerate(self.agents):
                    action, log_prob, value = agent.select_action(states[i])
                    actions[i] = action
                    log_probs.append(log_prob)
                    values.append(value)
                next_states, rewards, done, info = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    agent.store_transition(states[i], actions[i], log_probs[i], rewards[i], values[i], done)
                    ep_reward[i] += rewards[i]
                states = next_states
            for agent in self.agents:
                agent.finish_episode()
            if verbose and (ep + 1) % max(1, episodes // 10) == 0:
                avg_reward = ep_reward.mean()
                print(f"Episode {ep+1}/{episodes}, average reward per microgrid: {avg_reward:.3f}")