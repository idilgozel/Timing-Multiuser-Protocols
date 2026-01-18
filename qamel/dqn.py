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

class DQNNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.net(x)
