import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.agent import Agent
from qamel.environment import RepeaterChain
from qamel.utils import check_if_bad_state, check_if_final_state, compute_reward, get_episode_status, is_action_valid_given_state

import numpy as np
import torch
import argparse
import time
import random
import json
import csv
from collections import deque
import torch.nn as nn
import torch.optim as optim
from rich.progress import track

from qamel.utils import generate_all_valid_actions, linear_schedule
from qamel.dqn import DQNNet, build_dqn_net, preprocess_obs
from qamel.utils import chain_progress_potential_batch

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

SUPPORTED_OBS_MODES = {"baseline", "counter_exposed", "counter_exposed_plus_ready"}
DQN_OBS_MODES = {"counter_exposed", "counter_exposed_plus_ready"}
SUPPORTED_REWARD_MODES = {"base"}
METRICS_HEADERS = [
    "episode",
    "global_step",
    "epsilon",
    "avg_return_window",
    "success_proxy_window",
    "avg_steps_window",
    "avg_loss_window",
    "avg_ready_nodes_window",
    "checkpoint_saved",
]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        # Store terminated and truncated separately: only terminated masks the
        # bootstrap in the Bellman target; truncated transitions still bootstrap.
        self.buffer.append((obs, action, reward, next_obs, terminated, truncated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, terminated, truncated = zip(*batch)
        return obs, actions, rewards, next_obs, terminated, truncated

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {
            "capacity": self.buffer.maxlen,
            "items": list(self.buffer),
        }

    def load_state_dict(self, state_dict):
        capacity = state_dict.get("capacity", self.buffer.maxlen)
        items = state_dict.get("items", [])
        cpu_items = []
        for item in items:
            cpu_item = tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in item)
            # Migrate legacy 5-tuples (obs, a, r, next_obs, done): treat the old
            # done flag as terminated and truncated=False so loading never crashes.
            if len(cpu_item) == 5:
                cpu_item = (*cpu_item[:4], cpu_item[4], False)
            cpu_items.append(cpu_item)
        self.buffer = deque(cpu_items, maxlen=capacity)

def _as_serializable_hparams(hyperparameter_configs):
    return {
        key: getattr(hyperparameter_configs, key)
        for key in dir(hyperparameter_configs)
        if not key.startswith("_") and not callable(getattr(hyperparameter_configs, key))
    }

def _build_run_name(env_vars, model_tag):
    return f"dqn_n{env_vars['n']}_pgen{env_vars['pgen']}_pswap{env_vars['pswap']}_{model_tag}"

def _build_dqn_paths(env_vars, model_tag):
    run_name = _build_run_name(env_vars, model_tag)
    run_dir = os.path.join("qamel", "outputs", "runs", run_name)
    run_model_path = os.path.join(run_dir, "model.pt")
    config_path = os.path.join(run_dir, "config.json")
    train_log_path = os.path.join(run_dir, "train.log")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    model_path = os.path.join("qamel", "outputs", "models", f"{run_name}.pt")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    best_eval_checkpoint_path = os.path.join(checkpoint_dir, "best_eval.pt")
    best_eval_metrics_path = os.path.join(checkpoint_dir, "best_eval_metrics.json")
    legacy_checkpoint_dir = os.path.join("qamel", "outputs", "checkpoints", run_name)
    legacy_latest_checkpoint_path = os.path.join(legacy_checkpoint_dir, "latest.pt")
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "run_model_path": run_model_path,
        "config_path": config_path,
        "train_log_path": train_log_path,
        "metrics_path": metrics_path,
        "model_path": model_path,
        "checkpoint_dir": checkpoint_dir,
        "latest_checkpoint_path": latest_checkpoint_path,
        "best_eval_checkpoint_path": best_eval_checkpoint_path,
        "best_eval_metrics_path": best_eval_metrics_path,
        "legacy_checkpoint_dir": legacy_checkpoint_dir,
        "legacy_latest_checkpoint_path": legacy_latest_checkpoint_path,
    }

def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def _log_message(train_log_path, message):
    os.makedirs(os.path.dirname(train_log_path), exist_ok=True)
    with open(train_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{_timestamp()}] {message}\n")

def _ensure_metrics_file(metrics_path):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if os.path.exists(metrics_path):
        return
    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_HEADERS)
        writer.writeheader()

def _append_metrics_row(metrics_path, row):
    _ensure_metrics_file(metrics_path)
    with open(metrics_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRICS_HEADERS)
        writer.writerow(row)

def _write_run_config_if_missing(config_path, config_payload):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    if os.path.exists(config_path):
        return
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_payload, config_file, indent=2, sort_keys=True, default=_json_default)
        config_file.write("\n")

def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")

def _count_ready_nodes(state0):
    if state0.size(0) <= 2:
        return 0
    degrees = torch.count_nonzero(state0, dim=1)
    return int(torch.sum(degrees[1:-1] == 2).item())

def _safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ""
    return float(np.mean(finite))

def _set_global_seed(seed, torch_device):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def _capture_rng_state(torch_device):
    rng_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch_device.type == "cuda" and torch.cuda.is_available():
        rng_state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return rng_state

def _restore_rng_state(checkpoint, torch_device):
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        rng_state = checkpoint["torch_random_state"]
        if not isinstance(rng_state, torch.ByteTensor):
            rng_state = rng_state.cpu().byte()
        torch.random.set_rng_state(rng_state)
    if (
        torch_device.type == "cuda"
        and torch.cuda.is_available()
        and "torch_cuda_random_state_all" in checkpoint
    ):
        cuda_rng_states = checkpoint["torch_cuda_random_state_all"]
        cuda_rng_states = [
            s.cpu().byte() if not isinstance(s, torch.ByteTensor) else s
            for s in cuda_rng_states
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

def _build_dqn_checkpoint_payload(
    *,
    env_vars,
    model_tag,
    obs_mode,
    reward_mode,
    swap_ready_bonus,
    max_actions,
    training_episodes,
    use_curriculum,
    prefer_swap_when_ready_train,
    curriculum_steps,
    curriculum_boundaries,
    seed,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    hyperparameter_configs,
    input_shape,
    completed_episode_idx,
    global_step,
    torch_device,
    episode_returns,
    episode_steps,
    episode_successes,
    episode_avg_losses,
    episode_ready_means,
    net_arch="dqn",
    double_dqn=False,
    pbrs=False,
    pbrs_scale=1.0,
    best_eval_metrics=None,
):
    payload = {
        "checkpoint_version": 1,
        "model_tag": model_tag,
        "n": env_vars["n"],
        "pgen": env_vars["pgen"],
        "pswap": env_vars["pswap"],
        "obs_mode": obs_mode,
        "reward_mode": reward_mode,
        "swap_ready_bonus": swap_ready_bonus,
        "max_actions": max_actions,
        "training_episodes": training_episodes,
        "use_curriculum": use_curriculum,
        "prefer_swap_when_ready_train": prefer_swap_when_ready_train,
        "net_arch": net_arch,
        "double_dqn": double_dqn,
        "pbrs": pbrs,
        "pbrs_scale": pbrs_scale,
        "curriculum_steps": curriculum_steps,
        "curriculum_boundaries": curriculum_boundaries,
        "seed": seed,
        "model_state": policy_net.state_dict(),
        "target_model_state": target_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "replay_buffer_state": replay_buffer.state_dict(),
        "input_shape": input_shape,
        "counter_norm": hyperparameter_configs.counter_norm,
        "dqn_hyperparameters": _as_serializable_hparams(hyperparameter_configs),
        "episode": completed_episode_idx,
        "global_step": global_step,
        "episode_returns": episode_returns,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
        "episode_avg_losses": episode_avg_losses,
        "episode_ready_means": episode_ready_means,
        "best_eval_metrics": best_eval_metrics,
        "epsilon": linear_schedule(
            hyperparameter_configs.eps_start,
            hyperparameter_configs.eps_end,
            hyperparameter_configs.eps_decay_steps,
            global_step,
        ),
    }
    payload.update(_capture_rng_state(torch_device))
    return payload

def _save_checkpoint_payload(checkpoint_dir, payload, completed_episode_idx):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"episode_{completed_episode_idx + 1:06d}.pt"
    )
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(payload, checkpoint_path)
    torch.save(payload, latest_path)
    return checkpoint_path, latest_path

def _load_resume_checkpoint(model_path, latest_checkpoint_path, legacy_latest_checkpoint_path, torch_device):
    if os.path.exists(latest_checkpoint_path):
        return (
            torch.load(latest_checkpoint_path, map_location=torch_device, weights_only=False),
            latest_checkpoint_path,
        )
    if os.path.exists(legacy_latest_checkpoint_path):
        return (
            torch.load(legacy_latest_checkpoint_path, map_location=torch_device, weights_only=False),
            legacy_latest_checkpoint_path,
        )
    if os.path.exists(model_path):
        return torch.load(model_path, map_location=torch_device, weights_only=False), model_path
    return None, None

def _validate_resume_checkpoint(
    checkpoint,
    *,
    env_vars,
    obs_mode,
    reward_mode,
    swap_ready_bonus,
    prefer_swap_when_ready_train,
    seed,
    net_arch,
):
    if checkpoint is None:
        return
    expected = {
        "n": env_vars["n"],
        "pgen": env_vars["pgen"],
        "pswap": env_vars["pswap"],
        "obs_mode": obs_mode,
        "reward_mode": reward_mode,
        "swap_ready_bonus": swap_ready_bonus,
        "prefer_swap_when_ready_train": prefer_swap_when_ready_train,
        "seed": seed,
        "net_arch": net_arch,
    }
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        checkpoint_value = checkpoint.get(key)
        if checkpoint_value is not None and checkpoint_value != expected_value:
            raise ValueError(
                f"Checkpoint mismatch for '{key}': expected {expected_value}, found {checkpoint_value}."
            )

def _load_best_eval_metrics(best_eval_metrics_path, resume_checkpoint):
    if resume_checkpoint is not None and resume_checkpoint.get("best_eval_metrics") is not None:
        return resume_checkpoint.get("best_eval_metrics")
    if os.path.exists(best_eval_metrics_path):
        with open(best_eval_metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None

def _is_better_eval_metrics(candidate, current_best):
    if current_best is None:
        return True
    tol = 1e-12
    if candidate["success_rate"] > current_best["success_rate"] + tol:
        return True
    if candidate["success_rate"] < current_best["success_rate"] - tol:
        return False
    if candidate["mean_return"] > current_best["mean_return"] + tol:
        return True
    if candidate["mean_return"] < current_best["mean_return"] - tol:
        return False
    if candidate["mean_steps"] < current_best["mean_steps"] - tol:
        return True
    return False

# Cache of valid-action indices keyed by the binary occupancy pattern of state0.
# is_action_valid_given_state reads state0 only through (state0 == 0) / count_nonzero,
# so the cache returns bit-identical results -- it is a pure speedup, not a behaviour
# change. all_actions is fixed within a process, so it is not part of the key.
_VALID_INDEX_CACHE = {}

def _get_valid_action_indices(state0, all_actions):
    key = tuple((state0 != 0).detach().flatten().to(torch.uint8).cpu().tolist())
    cached = _VALID_INDEX_CACHE.get(key)
    if cached is not None:
        return list(cached)
    valid_indices = []
    for idx in range(all_actions.size(0)):
        if is_action_valid_given_state(state0, all_actions[idx]):
            valid_indices.append(idx)
    _VALID_INDEX_CACHE[key] = tuple(valid_indices)
    return valid_indices

def _filter_training_action_indices(
    state0,
    all_actions,
    valid_indices,
    prefer_swap_when_ready_train=False,
):
    if not prefer_swap_when_ready_train:
        return valid_indices
    if _count_ready_nodes(state0) <= 0:
        return valid_indices
    swap_indices = [idx for idx in valid_indices if _action_has_swap(all_actions[idx])]
    if swap_indices:
        return swap_indices
    return valid_indices

def _build_valid_action_mask(state0_batch, all_actions, no_op_idx=None):
    """Build a per-sample valid-action mask from a batch of adjacency matrices."""
    batch_size = state0_batch.size(0)
    num_actions = all_actions.size(0)
    valid_mask = torch.zeros(batch_size, num_actions, dtype=torch.bool, device=state0_batch.device)
    for batch_idx in range(batch_size):
        valid_indices = _get_valid_action_indices(state0_batch[batch_idx], all_actions)
        if len(valid_indices) == 0:
            if no_op_idx is None:
                raise ValueError("No valid actions available and no no-op action is defined.")
            valid_mask[batch_idx, no_op_idx] = True
            continue
        valid_mask[batch_idx, valid_indices] = True
    return valid_mask

def _action_has_swap(action_matrix):
    diag = torch.diagonal(action_matrix, 0)
    if diag.numel() > 2:
        diag = diag[1:-1]
    return bool((diag > 0).any().item())

def _evaluate_policy_snapshot(
    *,
    env_vars,
    policy_net,
    all_actions,
    obs_mode,
    counter_norm,
    max_actions,
    reward_mode,
    swap_ready_bonus,
    eval_episodes,
    eval_epsilon,
    torch_device,
    no_op_idx,
    seed,
):
    rng_state = _capture_rng_state(torch_device)
    try:
        _set_global_seed(seed, torch_device)
        eval_env = RepeaterChain(env_vars["n"], env_vars["pgen"], env_vars["pswap"], torch_device)
        was_training = policy_net.training
        policy_net.eval()

        rewards = []
        steps_list = []
        successes = []
        truncations = []
        ent_counts = []
        swap_counts = []

        for _ in range(eval_episodes):
            current_state = eval_env.reset()
            cumulative_reward = 0.0
            step = 0
            done = False
            while not done:
                obs = preprocess_obs(current_state, obs_mode, counter_norm).to(torch_device)
                valid_indices = _get_valid_action_indices(current_state[0], all_actions)
                if len(valid_indices) == 0:
                    action_idx = no_op_idx if no_op_idx is not None else 0
                elif eval_epsilon > 0.0 and random.random() < eval_epsilon:
                    action_idx = random.choice(valid_indices)
                else:
                    with torch.no_grad():
                        q_values = policy_net(obs.unsqueeze(0)).squeeze(0)
                        valid_mask = torch.zeros(all_actions.size(0), dtype=torch.bool, device=q_values.device)
                        valid_mask[valid_indices] = True
                        q_values[~valid_mask] = -1e9
                        action_idx = int(torch.argmax(q_values).item())

                new_state = eval_env.step(current_state, all_actions[action_idx])
                step += 1
                episode_status = get_episode_status(new_state, step, max_actions)
                reward = compute_reward(
                    new_state,
                    episode_status["final_state"],
                    episode_status["bad_state"] or episode_status["truncated"],
                    reward_mode=reward_mode,
                    swap_ready_bonus=swap_ready_bonus,
                    prev_state0=current_state[0],
                    action_matrix=all_actions[action_idx],
                )
                cumulative_reward += reward
                current_state = new_state
                done = episode_status["done"]

            rewards.append(float(cumulative_reward))
            steps_list.append(int(step))
            successes.append(float(episode_status["final_state"]))
            truncations.append(float(episode_status["truncated"]))
            ent_counts.append(float(torch.amax(current_state[1]).item()))
            swap_counts.append(float(torch.amax(current_state[2]).item()))

        metrics = {
            "success_rate": float(np.mean(successes)),
            "timeout_rate": float(np.mean(truncations)),
            "mean_return": float(np.mean(rewards)),
            "mean_steps": float(np.mean(steps_list)),
            "mean_ent_attempt_max": float(np.mean(ent_counts)),
            "mean_swap_attempt_max": float(np.mean(swap_counts)),
        }
        if was_training:
            policy_net.train()
        return metrics
    finally:
        _restore_rng_state(rng_state, torch_device)

def train_dqn_agent(env_vars, hyperparameter_configs, **kwargs):
    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    training_episodes = kwargs["training_episodes"]
    max_actions = kwargs["max_actions"]
    torch_device = kwargs["torch_device"]
    obs_mode = kwargs["obs_mode"]
    swap_ready_bonus = kwargs["swap_ready_bonus"]
    reward_mode = kwargs["reward_mode"]
    debug_target_mask = kwargs.get("debug_target_mask", False)
    checkpoint_dir = kwargs["checkpoint_dir"]
    checkpoint_every = kwargs["checkpoint_every"]
    log_every = kwargs["log_every"]
    model_tag = kwargs["model_tag"]
    train_log_path = kwargs["train_log_path"]
    metrics_path = kwargs["metrics_path"]
    seed = kwargs.get("seed")
    use_curriculum = kwargs["use_curriculum"]
    curriculum_steps = kwargs["curriculum_steps"]
    curriculum_boundaries = kwargs["curriculum_boundaries"]
    best_eval_every = kwargs["best_eval_every"]
    best_eval_episodes = kwargs["best_eval_episodes"]
    best_eval_max_actions = kwargs["best_eval_max_actions"]
    best_eval_seed = kwargs["best_eval_seed"]
    best_eval_metrics_path = kwargs["best_eval_metrics_path"]
    best_eval_checkpoint_path = kwargs["best_eval_checkpoint_path"]
    prefer_swap_when_ready_train = kwargs["prefer_swap_when_ready_train"]
    dueling = kwargs.get("dueling", False)
    double_dqn = kwargs.get("double_dqn", False)
    pbrs = kwargs.get("pbrs", False)
    pbrs_scale = kwargs.get("pbrs_scale", 1.0)
    net_arch = "dueling" if dueling else "dqn"

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
    no_op_candidates = (all_actions.view(num_actions, -1).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    no_op_idx = int(no_op_candidates[0].item()) if len(no_op_candidates) > 0 else None
    input_channels = 4 if obs_mode == "counter_exposed_plus_ready" else 3
    input_shape = (input_channels, n, n)

    policy_net = build_dqn_net(input_shape, num_actions, net_arch).to(torch_device)
    target_net = build_dqn_net(input_shape, num_actions, net_arch).to(torch_device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=hyperparameter_configs.lr)
    resume_checkpoint = kwargs.get("resume_checkpoint")
    replay_buffer = ReplayBuffer(hyperparameter_configs.buffer_size)
    start_episode = 0
    steps_done = 0
    if resume_checkpoint is not None:
        policy_net.load_state_dict(resume_checkpoint["model_state"])
        if "target_model_state" in resume_checkpoint:
            target_net.load_state_dict(resume_checkpoint["target_model_state"])
        else:
            target_net.load_state_dict(policy_net.state_dict())
        if "optimizer_state" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "replay_buffer_state" in resume_checkpoint:
            replay_buffer.load_state_dict(resume_checkpoint["replay_buffer_state"])
        start_episode = int(resume_checkpoint.get("episode", -1)) + 1
        steps_done = int(resume_checkpoint.get("global_step", 0))
        _restore_rng_state(resume_checkpoint, torch_device)
        print(
            f"Resuming from checkpoint after {start_episode} completed episode(s), "
            f"global_step={steps_done}, replay_size={len(replay_buffer)}"
        )
    else:
        target_net.load_state_dict(policy_net.state_dict())

    cumulative_reward_per_episode = np.zeros(training_episodes)
    episode_steps = np.zeros(training_episodes)
    episode_successes = np.zeros(training_episodes)
    episode_avg_losses = np.full(training_episodes, np.nan)
    episode_ready_means = np.zeros(training_episodes)
    if resume_checkpoint is not None and "episode_returns" in resume_checkpoint:
        saved_returns = np.asarray(resume_checkpoint["episode_returns"], dtype=float)
        copy_upto = min(len(saved_returns), training_episodes)
        cumulative_reward_per_episode[:copy_upto] = saved_returns[:copy_upto]
    if resume_checkpoint is not None and "episode_steps" in resume_checkpoint:
        saved_steps = np.asarray(resume_checkpoint["episode_steps"], dtype=float)
        episode_steps[: min(len(saved_steps), training_episodes)] = saved_steps[: min(len(saved_steps), training_episodes)]
    if resume_checkpoint is not None and "episode_successes" in resume_checkpoint:
        saved_successes = np.asarray(resume_checkpoint["episode_successes"], dtype=float)
        episode_successes[: min(len(saved_successes), training_episodes)] = saved_successes[: min(len(saved_successes), training_episodes)]
    if resume_checkpoint is not None and "episode_avg_losses" in resume_checkpoint:
        saved_losses = np.asarray(resume_checkpoint["episode_avg_losses"], dtype=float)
        episode_avg_losses[: min(len(saved_losses), training_episodes)] = saved_losses[: min(len(saved_losses), training_episodes)]
    if resume_checkpoint is not None and "episode_ready_means" in resume_checkpoint:
        saved_ready = np.asarray(resume_checkpoint["episode_ready_means"], dtype=float)
        episode_ready_means[: min(len(saved_ready), training_episodes)] = saved_ready[: min(len(saved_ready), training_episodes)]
    best_eval_metrics = _load_best_eval_metrics(best_eval_metrics_path, resume_checkpoint)
    target_mask_debug_printed = False

    if start_episode >= training_episodes:
        skip_message = (
            f"Checkpoint already covers requested training_episodes={training_episodes}. "
            "Skipping additional training."
        )
        print(skip_message)
        _log_message(train_log_path, skip_message)
        return (
            policy_net,
            target_net,
            optimizer,
            replay_buffer,
            cumulative_reward_per_episode,
            episode_steps,
            episode_successes,
            episode_avg_losses,
            episode_ready_means,
            steps_done,
            start_episode,
            best_eval_metrics,
        )

    current_limit = max_actions
    phase = None
    for episode in track(range(start_episode, training_episodes), description="Training DQN agent..."):
        if use_curriculum:
            if episode < curriculum_boundaries[0]:
                current_limit = curriculum_steps[0]
                if phase != 0:
                    print(f"Curriculum phase at episode {episode}: max_actions={current_limit}")
                    phase = 0
            elif episode < curriculum_boundaries[1]:
                current_limit = curriculum_steps[1]
                if phase != 1:
                    print(f"Curriculum phase at episode {episode}: max_actions={current_limit}")
                    phase = 1
            else:
                current_limit = curriculum_steps[2]
                if phase != 2:
                    print(f"Curriculum phase at episode {episode}: max_actions={current_limit}")
                    phase = 2
        cumulative_reward = 0
        step = 0
        done = False
        current_state = this_RepeaterChain.reset()
        episode_losses = []
        ready_nodes_total = 0.0
        ready_nodes_steps = 0

        while not done:
            epsilon = linear_schedule(
                hyperparameter_configs.eps_start,
                hyperparameter_configs.eps_end,
                hyperparameter_configs.eps_decay_steps,
                steps_done,
            )

            ready_nodes_total += _count_ready_nodes(current_state[0])
            ready_nodes_steps += 1
            obs = preprocess_obs(current_state, obs_mode, hyperparameter_configs.counter_norm).to(torch_device)
            valid_indices = _get_valid_action_indices(current_state[0], all_actions)
            selected_valid_indices = _filter_training_action_indices(
                current_state[0],
                all_actions,
                valid_indices,
                prefer_swap_when_ready_train=prefer_swap_when_ready_train,
            )
            if len(selected_valid_indices) == 0:
                action_idx = no_op_idx if no_op_idx is not None else random.randrange(num_actions)
            elif random.random() < epsilon:
                action_idx = random.choice(selected_valid_indices)
            else:
                with torch.no_grad():
                    q_values = policy_net(obs.unsqueeze(0)).squeeze(0)
                    valid_mask = torch.zeros(num_actions, dtype=torch.bool, device=q_values.device)
                    valid_mask[selected_valid_indices] = True
                    q_values[~valid_mask] = -1e9
                    action_idx = int(torch.argmax(q_values).item())

            new_state = this_RepeaterChain.step(current_state, all_actions[action_idx])
            step += 1
            steps_done += 1
            episode_status = get_episode_status(new_state, step, current_limit)
            reward = compute_reward(
                new_state,
                episode_status["final_state"],
                episode_status["bad_state"] or episode_status["truncated"],
                reward_mode=reward_mode,
                swap_ready_bonus=swap_ready_bonus,
                prev_state0=current_state[0],
                action_matrix=all_actions[action_idx],
            )

            # Potential-based reward shaping (Change 4). Shaping affects only the
            # replay target (r_train); episode-return logging stays unshaped (r_env)
            # so success rates and mean returns remain comparable across runs.
            r_train = reward
            if pbrs:
                phi_s = chain_progress_potential_batch(current_state.unsqueeze(0))[0].item()
                phi_s_next = chain_progress_potential_batch(new_state.unsqueeze(0))[0].item()
                shaping = hyperparameter_configs.gamma * phi_s_next - phi_s
                r_train = reward + pbrs_scale * shaping

            next_obs = preprocess_obs(new_state, obs_mode, hyperparameter_configs.counter_norm).to(torch_device)
            replay_buffer.add(
                obs.cpu(),
                action_idx,
                r_train,
                next_obs.cpu(),
                episode_status["terminated"],
                episode_status["truncated"],
            )

            current_state = new_state
            cumulative_reward += reward

            if len(replay_buffer) >= hyperparameter_configs.batch_size:
                obs_batch, actions_batch, rewards_batch, next_obs_batch, terminated_batch, truncated_batch = replay_buffer.sample(
                    hyperparameter_configs.batch_size
                )
                obs_batch = torch.stack(obs_batch).to(torch_device)
                actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=torch_device).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=torch_device).unsqueeze(1)
                next_obs_batch = torch.stack(next_obs_batch).to(torch_device)
                # Only `terminated` (success / bad state) masks the bootstrap; `truncated`
                # (hit max_actions) still bootstraps from V(s') -- the episode could continue.
                terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32, device=torch_device).unsqueeze(1)

                current_q = policy_net(obs_batch).gather(1, actions_batch)
                with torch.no_grad():
                    # Channel 0 remains the adjacency matrix for both DQN observation modes.
                    next_state0_batch = next_obs_batch[:, 0]
                    next_valid_mask = _build_valid_action_mask(next_state0_batch, all_actions, no_op_idx)
                    next_q_target_values = target_net(next_obs_batch)
                    if double_dqn:
                        # Double-DQN: policy net selects the (valid) argmax action,
                        # target net evaluates it.
                        next_q_policy = policy_net(next_obs_batch).masked_fill(~next_valid_mask, -1e9)
                        a_star = next_q_policy.argmax(dim=1, keepdim=True)
                        next_q = next_q_target_values.gather(1, a_star)
                        raw_next_q = next_q_target_values.max(1, keepdim=True)[0]
                    else:
                        masked_next_q_values = next_q_target_values.masked_fill(~next_valid_mask, -1e9)
                        raw_next_q = next_q_target_values.max(1, keepdim=True)[0]
                        next_q = masked_next_q_values.max(1, keepdim=True)[0]
                    if debug_target_mask and not target_mask_debug_printed:
                        sample_valid = int(next_valid_mask[0].sum().item())
                        print(
                            "Target-mask debug:",
                            f"raw_next_q={raw_next_q[0].item():.4f}",
                            f"masked_next_q={next_q[0].item():.4f}",
                            f"valid_actions={sample_valid}",
                        )
                        target_mask_debug_printed = True
                    target_q = rewards_batch + (1.0 - terminated_batch) * hyperparameter_configs.gamma * next_q

                loss = nn.SmoothL1Loss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                optimizer.step()
                episode_losses.append(float(loss.item()))

            if steps_done % hyperparameter_configs.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if episode_status["done"]:
                done = True
                cumulative_reward_per_episode[episode] = cumulative_reward
                episode_steps[episode] = step
                episode_successes[episode] = float(episode_status["final_state"])
                episode_avg_losses[episode] = (
                    float(np.mean(episode_losses)) if episode_losses else np.nan
                )
                episode_ready_means[episode] = (
                    ready_nodes_total / ready_nodes_steps if ready_nodes_steps > 0 else 0.0
                )
        completed_episodes = episode + 1
        checkpoint_saved = checkpoint_every > 0 and completed_episodes % checkpoint_every == 0
        should_log_metrics = log_every > 0 and (
            completed_episodes % log_every == 0 or completed_episodes == training_episodes
        )
        if should_log_metrics:
            window_start = max(0, completed_episodes - log_every)
            window_slice = slice(window_start, completed_episodes)
            epsilon_now = linear_schedule(
                hyperparameter_configs.eps_start,
                hyperparameter_configs.eps_end,
                hyperparameter_configs.eps_decay_steps,
                steps_done,
            )
            metrics_row = {
                "episode": completed_episodes,
                "global_step": steps_done,
                "epsilon": round(float(epsilon_now), 6),
                "avg_return_window": round(float(np.mean(cumulative_reward_per_episode[window_slice])), 6),
                "success_proxy_window": round(float(np.mean(episode_successes[window_slice])), 6),
                "avg_steps_window": round(float(np.mean(episode_steps[window_slice])), 6),
                "avg_loss_window": _safe_nanmean(episode_avg_losses[window_slice]),
                "avg_ready_nodes_window": round(float(np.mean(episode_ready_means[window_slice])), 6),
                "checkpoint_saved": int(checkpoint_saved),
            }
            _append_metrics_row(metrics_path, metrics_row)
            log_message = (
                f"episode={completed_episodes} global_step={steps_done} epsilon={metrics_row['epsilon']} "
                f"avg_return_window={metrics_row['avg_return_window']} "
                f"success_proxy_window={metrics_row['success_proxy_window']} "
                f"avg_steps_window={metrics_row['avg_steps_window']} "
                f"avg_loss_window={metrics_row['avg_loss_window']} "
                f"avg_ready_nodes_window={metrics_row['avg_ready_nodes_window']} "
                f"checkpoint_saved={metrics_row['checkpoint_saved']}"
            )
            _log_message(train_log_path, log_message)
        if checkpoint_every > 0 and completed_episodes % checkpoint_every == 0:
            checkpoint_payload = _build_dqn_checkpoint_payload(
                env_vars=env_vars,
                model_tag=model_tag,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                swap_ready_bonus=swap_ready_bonus,
                max_actions=max_actions,
                training_episodes=training_episodes,
                use_curriculum=use_curriculum,
                prefer_swap_when_ready_train=prefer_swap_when_ready_train,
                curriculum_steps=curriculum_steps,
                curriculum_boundaries=curriculum_boundaries,
                seed=seed,
                policy_net=policy_net,
                target_net=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                hyperparameter_configs=hyperparameter_configs,
                input_shape=input_shape,
                net_arch=net_arch,
                double_dqn=double_dqn,
                pbrs=pbrs,
                pbrs_scale=pbrs_scale,
                completed_episode_idx=episode,
                global_step=steps_done,
                torch_device=torch_device,
                episode_returns=cumulative_reward_per_episode[:completed_episodes].tolist(),
                episode_steps=episode_steps[:completed_episodes].tolist(),
                episode_successes=episode_successes[:completed_episodes].tolist(),
                episode_avg_losses=episode_avg_losses[:completed_episodes].tolist(),
                episode_ready_means=episode_ready_means[:completed_episodes].tolist(),
                best_eval_metrics=best_eval_metrics,
            )
            checkpoint_path, _ = _save_checkpoint_payload(checkpoint_dir, checkpoint_payload, episode)
            checkpoint_message = (
                f"Saved checkpoint: {checkpoint_path} "
                f"(episode={completed_episodes}, global_step={steps_done})"
            )
            print(checkpoint_message)
            _log_message(train_log_path, checkpoint_message)

        should_run_best_eval = (
            best_eval_every > 0
            and (completed_episodes % best_eval_every == 0 or completed_episodes == training_episodes)
        )
        if should_run_best_eval:
            candidate_metrics = _evaluate_policy_snapshot(
                env_vars=env_vars,
                policy_net=policy_net,
                all_actions=all_actions,
                obs_mode=obs_mode,
                counter_norm=hyperparameter_configs.counter_norm,
                max_actions=best_eval_max_actions,
                reward_mode=reward_mode,
                swap_ready_bonus=swap_ready_bonus,
                eval_episodes=best_eval_episodes,
                eval_epsilon=0.0,
                torch_device=torch_device,
                no_op_idx=no_op_idx,
                seed=best_eval_seed,
            )
            candidate_metrics.update(
                {
                    "episode": int(completed_episodes),
                    "global_step": int(steps_done),
                    "eval_epsilon": 0.0,
                    "eval_episodes": int(best_eval_episodes),
                    "max_actions": int(best_eval_max_actions),
                    "seed": best_eval_seed,
                    "obs_mode": obs_mode,
                    "reward_mode": reward_mode,
                    "model_tag": model_tag,
                    "selection_rule": "primary=success_rate secondary=mean_return tertiary=lower_mean_steps",
                }
            )
            best_eval_log_message = (
                f"Best-eval check: episode={completed_episodes} global_step={steps_done} "
                f"success_rate={candidate_metrics['success_rate']:.4f} "
                f"mean_return={candidate_metrics['mean_return']:.4f} "
                f"mean_steps={candidate_metrics['mean_steps']:.4f}"
            )
            print(best_eval_log_message)
            _log_message(train_log_path, best_eval_log_message)
            if _is_better_eval_metrics(candidate_metrics, best_eval_metrics):
                checkpoint_payload = _build_dqn_checkpoint_payload(
                    env_vars=env_vars,
                    model_tag=model_tag,
                    obs_mode=obs_mode,
                    reward_mode=reward_mode,
                    swap_ready_bonus=swap_ready_bonus,
                    max_actions=max_actions,
                    training_episodes=training_episodes,
                    use_curriculum=use_curriculum,
                    prefer_swap_when_ready_train=prefer_swap_when_ready_train,
                    curriculum_steps=curriculum_steps,
                    curriculum_boundaries=curriculum_boundaries,
                    seed=seed,
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    hyperparameter_configs=hyperparameter_configs,
                    input_shape=input_shape,
                    net_arch=net_arch,
                    double_dqn=double_dqn,
                    pbrs=pbrs,
                    pbrs_scale=pbrs_scale,
                    completed_episode_idx=episode,
                    global_step=steps_done,
                    torch_device=torch_device,
                    episode_returns=cumulative_reward_per_episode[:completed_episodes].tolist(),
                    episode_steps=episode_steps[:completed_episodes].tolist(),
                    episode_successes=episode_successes[:completed_episodes].tolist(),
                    episode_avg_losses=episode_avg_losses[:completed_episodes].tolist(),
                    episode_ready_means=episode_ready_means[:completed_episodes].tolist(),
                    best_eval_metrics=candidate_metrics,
                )
                torch.save(checkpoint_payload, best_eval_checkpoint_path)
                _write_json(best_eval_metrics_path, candidate_metrics)
                best_eval_metrics = candidate_metrics
                update_message = (
                    f"New best-eval checkpoint: {best_eval_checkpoint_path} "
                    f"(episode={completed_episodes}, success_rate={candidate_metrics['success_rate']:.4f}, "
                    f"mean_return={candidate_metrics['mean_return']:.4f}, "
                    f"mean_steps={candidate_metrics['mean_steps']:.4f})"
                )
                print(update_message)
                _log_message(train_log_path, update_message)

    return (
        policy_net,
        target_net,
        optimizer,
        replay_buffer,
        cumulative_reward_per_episode,
        episode_steps,
        episode_successes,
        episode_avg_losses,
        episode_ready_means,
        steps_done,
        training_episodes,
        best_eval_metrics,
    )

def _resolve_obs_mode(cli_obs_mode, model_tag):
    if cli_obs_mode is not None:
        return cli_obs_mode
    if model_tag in SUPPORTED_OBS_MODES:
        return model_tag
    raise ValueError(
        f"Unable to infer --obs_mode from model_tag='{model_tag}'. "
        f"Pass --obs_mode explicitly. Supported obs modes: {sorted(SUPPORTED_OBS_MODES)}"
    )
        
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
    parser.add_argument("--obs_mode", type=str, choices=["baseline", "counter_exposed", "counter_exposed_plus_ready"], default=None)
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--swap_ready_bonus", type=float, default=0.5)
    parser.add_argument("--reward_mode", type=str, choices=sorted(SUPPORTED_REWARD_MODES), default="base")
    parser.add_argument("--debug_target_mask", action="store_true")
    parser.add_argument("--eps_decay_steps", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--best_eval_every", type=int, default=0)
    parser.add_argument("--best_eval_episodes", type=int, default=500)
    parser.add_argument("--best_eval_max_actions", type=int, default=100)
    parser.add_argument("--best_eval_seed", type=int, default=None)
    parser.add_argument("--prefer_swap_when_ready_train", action="store_true")
    parser.add_argument("--double-dqn", dest="double_dqn", action="store_true", default=False)
    parser.add_argument("--dueling", action="store_true", default=False)
    parser.add_argument("--pbrs", action="store_true", default=False)
    parser.add_argument("--pbrs-scale", dest="pbrs_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--curriculum_steps", type=str, default="20,40,100")
    parser.add_argument("--curriculum_boundaries", type=str, default="5000,10000")

    env_vars_class = parser.parse_args()
    env_vars = env_vars_class.__dict__
    _set_global_seed(env_vars_class.seed, torch_device)

    this_hyperparameters = hyperparameters
    this_dqn_hyperparameters = dqn_hyperparameters
    if env_vars_class.eps_decay_steps is not None:
        this_dqn_hyperparameters.eps_decay_steps = env_vars_class.eps_decay_steps
    if env_vars_class.lr is not None:
        this_dqn_hyperparameters.lr = env_vars_class.lr
    if env_vars_class.batch_size is not None:
        this_dqn_hyperparameters.batch_size = env_vars_class.batch_size
    net_arch = "dueling" if env_vars_class.dueling else "dqn"

    agent_mode = _resolve_obs_mode(env_vars_class.obs_mode, env_vars["model_tag"])
    print(f"Training mode: {agent_mode}")
    print(f"Reward mode: {env_vars_class.reward_mode}")
    print(f"eps_decay_steps: {this_dqn_hyperparameters.eps_decay_steps}")
    print(f"Prefer swap when ready during training: {env_vars_class.prefer_swap_when_ready_train}")
    if agent_mode in DQN_OBS_MODES:
        train_episodes = env_vars_class.train_episodes or training_episodes
        run_artifacts = _build_dqn_paths(env_vars, env_vars["model_tag"])
        run_dir = run_artifacts["run_dir"]
        model_path = run_artifacts["model_path"]
        run_model_path = run_artifacts["run_model_path"]
        checkpoint_dir = run_artifacts["checkpoint_dir"]
        latest_checkpoint_path = run_artifacts["latest_checkpoint_path"]
        legacy_latest_checkpoint_path = run_artifacts["legacy_latest_checkpoint_path"]
        train_log_path = run_artifacts["train_log_path"]
        metrics_path = run_artifacts["metrics_path"]
        config_path = run_artifacts["config_path"]
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if env_vars_class.force_train and any(os.scandir(run_dir)):
            warning_message = (
                "force_train=True with an existing run directory will append train.log/metrics.csv "
                "while overwriting the run-local model and latest checkpoint."
            )
            print(warning_message)
            _log_message(train_log_path, warning_message)
        run_config = {
            "created_at": _timestamp(),
            "run_name": run_artifacts["run_name"],
            "paths": run_artifacts,
            "env": {
                "n": env_vars["n"],
                "pgen": env_vars["pgen"],
                "pswap": env_vars["pswap"],
            },
            "train": {
                "model_tag": env_vars["model_tag"],
                "obs_mode": agent_mode,
                "seed": env_vars_class.seed,
                "reward_mode": env_vars_class.reward_mode,
                "swap_ready_bonus": env_vars_class.swap_ready_bonus,
                "train_episodes": train_episodes,
                "max_actions": max_actions,
                "checkpoint_every": env_vars_class.checkpoint_every,
                "log_every": env_vars_class.log_every,
                "best_eval_every": env_vars_class.best_eval_every,
                "best_eval_episodes": env_vars_class.best_eval_episodes,
                "best_eval_max_actions": env_vars_class.best_eval_max_actions,
                "best_eval_seed": env_vars_class.best_eval_seed,
                "prefer_swap_when_ready_train": env_vars_class.prefer_swap_when_ready_train,
                "net_arch": net_arch,
                "double_dqn": env_vars_class.double_dqn,
                "pbrs": env_vars_class.pbrs,
                "pbrs_scale": env_vars_class.pbrs_scale,
                "lr": this_dqn_hyperparameters.lr,
                "batch_size": this_dqn_hyperparameters.batch_size,
                "use_curriculum": env_vars_class.use_curriculum,
                "curriculum_steps": env_vars_class.curriculum_steps,
                "curriculum_boundaries": env_vars_class.curriculum_boundaries,
                "force_train": env_vars_class.force_train,
            },
            "dqn_hyperparameters": _as_serializable_hparams(this_dqn_hyperparameters),
        }
        if env_vars_class.force_train:
            _write_json(config_path, run_config)
        else:
            _write_run_config_if_missing(config_path, run_config)
        _ensure_metrics_file(metrics_path)
        _log_message(
            train_log_path,
            f"Starting training invocation force_train={env_vars_class.force_train} "
            f"seed={env_vars_class.seed} "
            f"prefer_swap_when_ready_train={env_vars_class.prefer_swap_when_ready_train}",
        )
        resume_checkpoint = None
        if not env_vars_class.force_train:
            resume_checkpoint, resume_path = _load_resume_checkpoint(
                model_path, latest_checkpoint_path, legacy_latest_checkpoint_path, torch_device
            )
            _validate_resume_checkpoint(
                resume_checkpoint,
                env_vars=env_vars,
                obs_mode=agent_mode,
                reward_mode=env_vars_class.reward_mode,
                swap_ready_bonus=env_vars_class.swap_ready_bonus,
                prefer_swap_when_ready_train=env_vars_class.prefer_swap_when_ready_train,
                seed=env_vars_class.seed,
                net_arch=net_arch,
            )
            if resume_checkpoint is not None:
                resume_message = f"Loaded DQN checkpoint from {resume_path}"
                print(resume_message)
                _log_message(train_log_path, resume_message)

        start_time = time.time()
        curriculum_steps = [int(x) for x in env_vars_class.curriculum_steps.split(",")]
        curriculum_boundaries = [int(x) for x in env_vars_class.curriculum_boundaries.split(",")]

        (
            model,
            target_net,
            model_optimizer,
            replay_buffer,
            training_rewards,
            episode_steps,
            episode_successes,
            episode_avg_losses,
            episode_ready_means,
            steps_done,
            completed_episodes,
            best_eval_metrics,
        ) = train_dqn_agent(
            env_vars,
            this_dqn_hyperparameters,
            max_actions=max_actions,
            training_episodes=train_episodes,
            torch_device=torch_device,
            obs_mode=agent_mode,
            model_tag=env_vars["model_tag"],
            resume_checkpoint=resume_checkpoint,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=env_vars_class.checkpoint_every,
            log_every=env_vars_class.log_every,
            best_eval_every=env_vars_class.best_eval_every,
            best_eval_episodes=env_vars_class.best_eval_episodes,
            best_eval_max_actions=env_vars_class.best_eval_max_actions,
            best_eval_seed=env_vars_class.best_eval_seed if env_vars_class.best_eval_seed is not None else env_vars_class.seed,
            best_eval_metrics_path=run_artifacts["best_eval_metrics_path"],
            best_eval_checkpoint_path=run_artifacts["best_eval_checkpoint_path"],
            prefer_swap_when_ready_train=env_vars_class.prefer_swap_when_ready_train,
            dueling=env_vars_class.dueling,
            double_dqn=env_vars_class.double_dqn,
            pbrs=env_vars_class.pbrs,
            pbrs_scale=env_vars_class.pbrs_scale,
            train_log_path=train_log_path,
            metrics_path=metrics_path,
            swap_ready_bonus=env_vars_class.swap_ready_bonus,
            reward_mode=env_vars_class.reward_mode,
            seed=env_vars_class.seed,
            debug_target_mask=env_vars_class.debug_target_mask,
            use_curriculum=env_vars_class.use_curriculum,
            curriculum_steps=curriculum_steps,
            curriculum_boundaries=curriculum_boundaries,
        )

        os.makedirs("qamel/outputs/models", exist_ok=True)
        input_channels = 4 if agent_mode == "counter_exposed_plus_ready" else 3
        final_payload = _build_dqn_checkpoint_payload(
            env_vars=env_vars,
            model_tag=env_vars["model_tag"],
            obs_mode=agent_mode,
            reward_mode=env_vars_class.reward_mode,
            swap_ready_bonus=env_vars_class.swap_ready_bonus,
            max_actions=max_actions,
            training_episodes=train_episodes,
            use_curriculum=env_vars_class.use_curriculum,
            prefer_swap_when_ready_train=env_vars_class.prefer_swap_when_ready_train,
            curriculum_steps=curriculum_steps,
            curriculum_boundaries=curriculum_boundaries,
            seed=env_vars_class.seed,
            policy_net=model,
            target_net=target_net,
            optimizer=model_optimizer,
            replay_buffer=replay_buffer,
            hyperparameter_configs=this_dqn_hyperparameters,
            input_shape=(input_channels, env_vars["n"], env_vars["n"]),
            net_arch=net_arch,
            double_dqn=env_vars_class.double_dqn,
            pbrs=env_vars_class.pbrs,
            pbrs_scale=env_vars_class.pbrs_scale,
            completed_episode_idx=max(0, completed_episodes - 1),
            global_step=steps_done,
            torch_device=torch_device,
            episode_returns=training_rewards[:completed_episodes].tolist(),
            episode_steps=episode_steps[:completed_episodes].tolist(),
            episode_successes=episode_successes[:completed_episodes].tolist(),
            episode_avg_losses=episode_avg_losses[:completed_episodes].tolist(),
            episode_ready_means=episode_ready_means[:completed_episodes].tolist(),
            best_eval_metrics=best_eval_metrics,
        )
        torch.save(final_payload, model_path)
        torch.save(final_payload, run_model_path)
        checkpoint_path, latest_path = _save_checkpoint_payload(
            checkpoint_dir,
            final_payload,
            max(0, completed_episodes - 1),
        )
        final_model_message = f"Saved final model to {model_path}"
        run_model_message = f"Saved run-local model to {run_model_path}"
        latest_checkpoint_message = f"Updated latest checkpoint at {latest_path}"
        print(final_model_message)
        print(run_model_message)
        print(latest_checkpoint_message)
        _log_message(train_log_path, final_model_message)
        _log_message(train_log_path, run_model_message)
        _log_message(train_log_path, latest_checkpoint_message)

        end_time = time.time()
        duration_message = f"Took {end_time - start_time} seconds to train."
        print(duration_message)
        _log_message(train_log_path, duration_message)

    else:
        train_episodes = env_vars_class.train_episodes or training_episodes
        if os.path.exists(f"qamel/q_table_storage/{env_vars['n']}_nodes.txt"):
            print(f"An agent for {env_vars['n']} nodes has been trained.")
        else:
            start_time = time.time()
            q_table, training_rewards = train_q_agent(env_vars, this_hyperparameters, max_actions = max_actions, training_episodes = train_episodes, torch_device = torch_device)

            os.makedirs("qamel/q_table_storage", exist_ok=True)
            np.savetxt(f"qamel/q_table_storage/{env_vars['n']}_nodes.txt", q_table.cpu().numpy())

            end_time = time.time()
            print(f"Took {end_time - start_time} seconds to train.")
