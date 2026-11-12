"""DQN agent (Mnih et al., 2015): Q-network, experience replay and target network.

Q-network: MLP 22 -> 64 -> 64 -> 5 (one Q-value per action), ReLU activations.
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores (state, action, reward, next_state, done) and samples random batches,
    which breaks the correlation between consecutive steps."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (np.array(obs), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_obs), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.95, weight_decay=0.0,
                 batch_size=64, buffer_size=20_000, device="cpu"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.q = QNetwork(obs_dim, n_actions).to(device)
        self.target_q = QNetwork(obs_dim, n_actions).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr, weight_decay=weight_decay)
        self.buffer = ReplayBuffer(buffer_size)

    @torch.no_grad()
    def act(self, obs, epsilon=0.0):
        """Epsilon-greedy: random action with probability epsilon, else the best Q-value."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q_values = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        return int(q_values.argmax())

    def update(self):
        """One gradient step on a random batch from the replay buffer."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)

        q_values = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():  # target: r + gamma * max_a' Q_target(s', a')
            target = rewards + self.gamma * (1 - dones) * self.target_q(next_obs).max(1).values
        loss = nn.functional.smooth_l1_loss(q_values, target)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)  # gradient clipping
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        """Copy the Q-network weights into the target network."""
        self.target_q.load_state_dict(self.q.state_dict())
