import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import numpy as np

from .utils import generate_all_valid_actions  # we keep action enumeration


class DQNNetwork(nn.Module):
    """
    Simple MLP that takes flattened state (3 * n * n) and outputs Q-values
    for each discrete action in self.all_actions.
    """
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        # x: (batch, state_dim)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        # Store everything as CPU tensors so sampling is easy
        self.buffer.append((
            s.detach().cpu(),
            int(a),
            float(r),
            s_next.detach().cpu(),
            bool(done),
        ))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.stack(s).to(device)                       # (B, state_dim)
        a = torch.tensor(a, dtype=torch.long, device=device)  # (B,)
        r = torch.tensor(r, dtype=torch.float32, device=device)  # (B,)
        s_next = torch.stack(s_next).to(device)             # (B, state_dim)
        done = torch.tensor(done, dtype=torch.float32, device=device)  # (B,)

        return s, a, r, s_next, done


class DQNAgent:
    """
    Deep Q-Learning agent for the repeater chain.

    Key differences from the tabular Agent:
    - No state enumeration.
    - State is treated as a continuous tensor (3, n, n) -> flattened vector.
    - Q(s, a) approximated by a neural network.
    """

    def __init__(
        self,
        n,
        device,
        gamma=0.99,
        lr=1e-3,
        hidden_dim=256,
        replay_capacity=200_000,
        batch_size=64,
        target_update_freq=1000,
    ):
        self.n = n
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # ----- Actions: we keep Anuj's generate_all_valid_actions -----
        self.all_actions = generate_all_valid_actions(self.n).to(self.device)
        self.n_actions = self.all_actions.size(0)

        # ----- State dimension: 3 layers of n x n -----
        self.state_dim = 3 * n * n

        # ----- Networks -----
        self.policy_net = DQNNetwork(self.state_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # ----- Replay buffer -----
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.learn_step_counter = 0

    # ---------- State encoding ----------

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (3, n, n) tensor on any device
        returns: flattened (state_dim,) tensor on self.device
        """
        # ensure float32 and correct device
        return state.to(self.device, dtype=torch.float32).view(-1)

    # ---------- Action selection ----------

    def predict_action(self, state: torch.Tensor, epsilon: float):
        """
        Epsilon-greedy policy.

        Returns:
        - action_idx (int): index into self.all_actions
        - action_matrix (tensor): (n, n) tensor that can be passed to env.step
        """
        state_vec = self.encode_state(state).unsqueeze(0)  # (1, state_dim)

        if random.random() < epsilon:
            # explore
            action_idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                q_vals = self.policy_net(state_vec)  # (1, n_actions)
                action_idx = int(torch.argmax(q_vals, dim=1).item())

        action_matrix = self.all_actions[action_idx]
        return action_idx, action_matrix

    # ---------- Experience + learning ----------

    def store_transition(self, s, a_idx, r, s_next, done):
        """
        s, s_next: (state_dim,) vectors (already encoded)
        a_idx: int
        """
        self.replay_buffer.push(s, a_idx, r, s_next, done)

    def update(self):
        """
        One gradient step of DQN if enough samples in replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size, self.device)

        # Q(s, a)
        q_vals = self.policy_net(s)                         # (B, n_actions)
        q_sa = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r + gamma * max_a' Q_target(s', a')   (0 if done)
        with torch.no_grad():
            q_next = self.target_net(s_next)                # (B, n_actions)
            max_q_next, _ = q_next.max(dim=1)               # (B,)
            target = r + self.gamma * (1.0 - done) * max_q_next

        loss = nn.MSELoss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
