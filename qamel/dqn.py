import torch
import torch.nn as nn

def preprocess_obs(state, obs_mode, counter_norm=20.0):
    if obs_mode == "counter_exposed":
        obs = state.clone().float()
        if counter_norm is not None and counter_norm > 0:
            obs[1:] = torch.clamp(obs[1:] / counter_norm, 0.0, 1.0)
        return obs
    return state[0].float()

class DQNNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        channels, height, width = input_shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x)
