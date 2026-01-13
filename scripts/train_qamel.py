import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.agent import Agent
from qamel.environment import RepeaterChain
from qamel.utils import check_if_bad_state, check_if_final_state, reward_shape

import numpy as np
import torch
import argparse
import time
import random
from collections import deque
import torch.nn as nn
import torch.optim as optim
from rich.progress import track

from qamel.utils import generate_all_valid_actions, linear_schedule
from qamel.dqn import DQNNet, preprocess_obs

def train_q_agent(env_vars, hyperparameter_configs, **kwargs):

    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    training_episodes = kwargs["training_episodes"]
    max_actions = kwargs["max_actions"]

    alpha = hyperparameter_configs.alpha
    gamma = hyperparameter_configs.gamma
    epsilon = hyperparameter_configs.epsilon

    torch_device = kwargs["torch_device"]

    this_QAlgorithm = Agent(n, alpha = alpha, gamma = gamma, device = torch_device)
    this_RepeaterChain = RepeaterChain(n, pgen, pswap, torch_device)

    cumulative_reward_per_episode = np.zeros(training_episodes)

    for episode in track(range(training_episodes), description="Training q-learning agent..."):
        cumulative_reward = 0; step = 0
        done = False
        current_state = this_RepeaterChain.reset()

        while not done:
            action = this_QAlgorithm.predict_action(current_state[0], epsilon)
            new_state = this_RepeaterChain.step(current_state, action)

            bad_state = check_if_bad_state(new_state)
            final_state = check_if_final_state(new_state)
            
            reward = reward_shape(new_state, final_state, bad_state)

            this_QAlgorithm.update_q_table(current_state[0], action, reward, new_state[0], bad_state)

            current_state = new_state

            step += 1
            cumulative_reward += reward

            if (step >= max_actions or check_if_final_state(current_state) or check_if_bad_state(current_state)):
                done = True
                cumulative_reward_per_episode[episode] = cumulative_reward

    return this_QAlgorithm.q_table, cumulative_reward_per_episode


class hyperparameters:
    alpha = 0.0755410045582013
    epsilon = 0.10210439965486162
    gamma = 0.027548998387277125

class dqn_hyperparameters:
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    buffer_size = 50000
    target_update_steps = 1000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 10000
    counter_norm = 20.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

def train_dqn_agent(env_vars, hyperparameter_configs, **kwargs):
    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    training_episodes = kwargs["training_episodes"]
    max_actions = kwargs["max_actions"]
    torch_device = kwargs["torch_device"]
    obs_mode = kwargs["obs_mode"]

    this_RepeaterChain = RepeaterChain(n, pgen, pswap, torch_device)

    actions_dir = os.path.join(os.path.dirname(__file__), "..", "qamel", "outputs", "logs", "actions")
    actions_path = os.path.join(actions_dir, f"{n}_nodes.npy")
    if os.path.exists(actions_path):
        all_actions = np.load(actions_path)
        all_actions = torch.Tensor(all_actions).to(torch_device)
    else:
        all_actions = generate_all_valid_actions(n).to(torch_device)
        os.makedirs(actions_dir, exist_ok=True)
        np.save(actions_path, all_actions.cpu().numpy())

    num_actions = all_actions.size(0)
    input_shape = (3, n, n)

    policy_net = DQNNet(input_shape, num_actions).to(torch_device)
    target_net = DQNNet(input_shape, num_actions).to(torch_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=hyperparameter_configs.lr)
    replay_buffer = ReplayBuffer(hyperparameter_configs.buffer_size)

    cumulative_reward_per_episode = np.zeros(training_episodes)
    steps_done = 0

    for episode in track(range(training_episodes), description="Training DQN agent..."):
        cumulative_reward = 0
        step = 0
        done = False
        current_state = this_RepeaterChain.reset()

        while not done:
            epsilon = linear_schedule(
                hyperparameter_configs.eps_start,
                hyperparameter_configs.eps_end,
                hyperparameter_configs.eps_decay_steps,
                steps_done,
            )

            obs = preprocess_obs(current_state, obs_mode, hyperparameter_configs.counter_norm).to(torch_device)
            if random.random() < epsilon:
                action_idx = random.randrange(num_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(obs.unsqueeze(0))
                    action_idx = int(torch.argmax(q_values, dim=1).item())

            new_state = this_RepeaterChain.step(current_state, all_actions[action_idx])

            bad_state = check_if_bad_state(new_state)
            final_state = check_if_final_state(new_state)
            reward = reward_shape(new_state, final_state, bad_state)

            next_obs = preprocess_obs(new_state, obs_mode, hyperparameter_configs.counter_norm).to(torch_device)
            replay_buffer.add(obs.cpu(), action_idx, reward, next_obs.cpu(), bad_state or final_state)

            current_state = new_state
            step += 1
            steps_done += 1
            cumulative_reward += reward

            if len(replay_buffer) >= hyperparameter_configs.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = replay_buffer.sample(
                    hyperparameter_configs.batch_size
                )
                obs_batch = torch.stack(obs_batch).to(torch_device)
                actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=torch_device).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=torch_device).unsqueeze(1)
                next_obs_batch = torch.stack(next_obs_batch).to(torch_device)
                dones_batch = torch.tensor(dones_batch, dtype=torch.float32, device=torch_device).unsqueeze(1)

                current_q = policy_net(obs_batch).gather(1, actions_batch)
                with torch.no_grad():
                    next_q = target_net(next_obs_batch).max(1, keepdim=True)[0]
                    target_q = rewards_batch + (1.0 - dones_batch) * hyperparameter_configs.gamma * next_q

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % hyperparameter_configs.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if (step >= max_actions or final_state or bad_state):
                done = True
                cumulative_reward_per_episode[episode] = cumulative_reward

    return policy_net, cumulative_reward_per_episode
        
max_actions = 100
training_episodes = 10000


if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", torch_device)

    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type = int)
    parser.add_argument("--pgen", type = float)
    parser.add_argument("--pswap", type = float)
    parser.add_argument("--model_tag", type=str, default="baseline")

    env_vars_class = parser.parse_args()
    env_vars = env_vars_class.__dict__

    this_hyperparameters = hyperparameters
    this_dqn_hyperparameters = dqn_hyperparameters

    if env_vars["model_tag"] == "counter_exposed":
        model_path = (
            f"qamel/outputs/models/dqn_n{env_vars['n']}_pgen{env_vars['pgen']}_pswap{env_vars['pswap']}_{env_vars['model_tag']}.pt"
        )
        if os.path.exists(model_path):
            print(f"A DQN agent for {env_vars['n']} nodes has been trained.")
        else:
            start_time = time.time()
            model, training_rewards = train_dqn_agent(
                env_vars,
                this_dqn_hyperparameters,
                max_actions=max_actions,
                training_episodes=training_episodes,
                torch_device=torch_device,
                obs_mode="counter_exposed",
            )

            os.makedirs("qamel/outputs/models", exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_shape": (3, env_vars["n"], env_vars["n"]),
                    "counter_norm": this_dqn_hyperparameters.counter_norm,
                },
                model_path,
            )

            end_time = time.time()
            print(f"Took {end_time - start_time} seconds to train.")

    else:
        if os.path.exists(f"qamel/q_table_storage/{env_vars['n']}_nodes.txt"):
            print(f"An agent for {env_vars['n']} nodes has been trained.")
        else:
            start_time = time.time()
            q_table, training_rewards = train_q_agent(env_vars, this_hyperparameters, max_actions = max_actions, training_episodes = training_episodes, torch_device = torch_device)

            os.makedirs("qamel/q_table_storage", exist_ok=True)
            np.savetxt(f"qamel/q_table_storage/{env_vars['n']}_nodes.txt", q_table.cpu().numpy())

            end_time = time.time()
            print(f"Took {end_time - start_time} seconds to train.")
