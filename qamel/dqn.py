import torch
import torch.nn as nn

def preprocess_obs(state, obs_mode, counter_norm=20.0):
    if obs_mode in ["counter_exposed", "counter_exposed_plus_ready"]:
        obs = state.clone().float()
        if counter_norm is not None and counter_norm > 0:
            obs[1:] = torch.clamp(obs[1:] / counter_norm, 0.0, 1.0)
        if obs_mode == "counter_exposed_plus_ready":
            degrees = torch.count_nonzero(state[0], dim=1).float()
            ready_nodes = torch.sum(degrees[1:-1] == 2).float()
            # Append readiness scalar as a constant channel to give global swap-readiness signal.
            ready_channel = torch.full_like(state[0], ready_nodes)
            obs = torch.cat([obs, ready_channel.unsqueeze(0)], dim=0)
        return obs
    return state[0].float()

def _hidden_size(input_size):
    # Shared width formula so DQNNet and DuelingDQNNet have predictable checkpoint shapes.
    return max(512, ((input_size * 4 + 255) // 256) * 256)

class DQNNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape
        input_size = channels * height * width
        hidden_size = _hidden_size(input_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)

class DuelingDQNNet(nn.Module):
    """Dueling DQN head: shared trunk feeding separate value and advantage heads.

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)). Uses the same hidden-width
    formula as DQNNet so checkpoint shapes stay predictable.
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape
        input_size = channels * height * width
        hidden_size = _hidden_size(input_size)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        features = self.net(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

def build_dqn_net(input_shape, num_actions, net_arch="dqn"):
    """Factory selecting the Q-network architecture by name.

    net_arch defaults to "dqn" so legacy checkpoints (which carry no net_arch
    field) reconstruct the original DQNNet and behave bit-for-bit identically.
    """
    if net_arch in (None, "dqn"):
        return DQNNet(input_shape, num_actions)
    if net_arch == "dueling":
        return DuelingDQNNet(input_shape, num_actions)
    raise ValueError(f"Unknown net_arch='{net_arch}'. Expected 'dqn' or 'dueling'.")
