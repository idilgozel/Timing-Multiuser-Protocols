import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.environment import RepeaterChain
from qamel.utils import compute_reward, get_episode_status, get_state_id, generate_all_valid_actions, is_action_valid_given_state

import numpy as np
import torch
import argparse
import time
from rich.progress import track
import csv
import json
import random

from qamel.dqn import DQNNet, build_dqn_net, preprocess_obs

SUPPORTED_OBS_MODES = {"baseline", "counter_exposed", "counter_exposed_plus_ready"}
DQN_OBS_MODES = {"counter_exposed", "counter_exposed_plus_ready"}
SUPPORTED_REWARD_MODES = {"base"}

def _select_observation(state, obs_mode, counter_norm=None):
    return preprocess_obs(state, obs_mode, counter_norm)

def _get_valid_action_indices(state0, all_actions):
    valid_indices = []
    for idx in range(all_actions.size(0)):
        if is_action_valid_given_state(state0, all_actions[idx]):
            valid_indices.append(idx)
    return valid_indices

def _action_requests_refresh(state0, action_matrix):
    active_edges = torch.triu(state0, diagonal=1) > 0
    requested_edges = torch.triu(action_matrix, diagonal=1) > 0
    return bool((active_edges & requested_edges).any().item())

def _filter_eval_action_indices(
    state0,
    all_actions,
    valid_indices,
    no_op_idx=None,
    mask_null_action=False,
    block_refresh_actions=False,
    prefer_swap_when_ready=False,
):
    filtered_indices = []
    for idx in valid_indices:
        if mask_null_action and no_op_idx is not None and idx == no_op_idx:
            continue
        if block_refresh_actions and _action_requests_refresh(state0, all_actions[idx]):
            continue
        filtered_indices.append(idx)

    if prefer_swap_when_ready and _count_swap_ready_nodes_from_state0(state0) > 0:
        swap_filtered = [idx for idx in filtered_indices if _action_has_swap(all_actions[idx])]
        if swap_filtered:
            filtered_indices = swap_filtered

    return filtered_indices

def _action_has_swap(action_matrix):
    diag = torch.diagonal(action_matrix, 0)
    if diag.numel() > 2:
        diag = diag[1:-1]
    return bool((diag > 0).any().item())

def _count_swap_ready_nodes_from_state0(state0):
    degrees = torch.count_nonzero(state0, dim=1)
    if state0.size(0) <= 2:
        return 0
    return int(torch.sum(degrees[1:-1] == 2).item())

def _count_active_links(state0):
    return int(torch.count_nonzero(torch.triu(state0, diagonal=1)).item())

def _active_edge_list(state0):
    upper_idx = (torch.triu(state0, diagonal=1) > 0).nonzero(as_tuple=False)
    return [(int(i.item()), int(j.item())) for i, j in upper_idx]

def _summarize_state(state):
    state0 = state[0]
    return {
        "active_links": _count_active_links(state0),
        "active_edge_list": _active_edge_list(state0),
        "swap_ready_nodes": _count_swap_ready_nodes_from_state0(state0),
        "ent_attempt_max": float(torch.amax(state[1]).item()),
        "swap_attempt_max": float(torch.amax(state[2]).item()),
    }

def _summarize_action(action_matrix, state0=None, action_id=None, null_action_idx=None):
    upper_idx = (torch.triu(action_matrix, diagonal=1) > 0).nonzero(as_tuple=False)
    generation_edges = [(int(i.item()), int(j.item())) for i, j in upper_idx]
    diag = torch.diagonal(action_matrix, 0)
    swap_nodes = []
    if diag.numel() > 2:
        swap_nodes = [int(idx.item()) + 1 for idx in (diag[1:-1] > 0).nonzero(as_tuple=False).flatten()]
    refresh_edges = []
    if state0 is not None:
        refresh_edges = [(i, j) for (i, j) in generation_edges if float(state0[i, j].item()) > 0]
    return {
        "action_index": int(action_id) if action_id is not None else None,
        "is_null_action": bool(action_id == null_action_idx) if null_action_idx is not None and action_id is not None else len(generation_edges) == 0 and len(swap_nodes) == 0,
        "num_generation_requests": len(generation_edges),
        "num_swap_requests": len(swap_nodes),
        "generation_edges": generation_edges,
        "swap_nodes": swap_nodes,
        "refresh_edges": refresh_edges,
    }

def _resolve_obs_mode(cli_obs_mode, model_tag):
    if cli_obs_mode is not None:
        return cli_obs_mode
    if model_tag in SUPPORTED_OBS_MODES:
        return model_tag
    raise ValueError(
        f"Unable to infer --obs_mode from model_tag='{model_tag}'. "
        f"Pass --obs_mode explicitly. Supported obs modes: {sorted(SUPPORTED_OBS_MODES)}"
    )

def _build_run_name(n, pgen, pswap, model_tag):
    return f"dqn_n{n}_pgen{pgen}_pswap{pswap}_{model_tag}"

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

def _build_evaluation_artifact_paths(n, pgen, pswap, model_tag, eval_episodes, max_actions, eval_epsilon, seed):
    run_name = _build_run_name(n, pgen, pswap, model_tag)
    run_dir = os.path.join("qamel", "outputs", "runs", run_name)
    run_model_path = os.path.join(run_dir, "model.pt")
    evaluations_dir = os.path.join(run_dir, "evaluations")
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    eval_name = f"eval_episodes{eval_episodes}_max{max_actions}_eps{eval_epsilon}{seed_suffix}"
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "run_model_path": run_model_path,
        "evaluations_dir": evaluations_dir,
        "csv_path": os.path.join(evaluations_dir, f"{eval_name}.csv"),
        "summary_path": os.path.join(evaluations_dir, f"{eval_name}_summary.json"),
        "ent_counts_path": os.path.join(evaluations_dir, f"{eval_name}_ent_counts.txt"),
        "swap_counts_path": os.path.join(evaluations_dir, f"{eval_name}_swap_counts.txt"),
    }

def evaluate_q_table(env_vars, **kwargs):

    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    eval_episodes = kwargs["eval_episodes"]

    torch_device = kwargs["torch_device"]
    obs_mode = kwargs["obs_mode"]
    states_path = kwargs["states_path"]
    q_table_path = kwargs["q_table_path"]
    model_path = kwargs["model_path"]
    counter_norm = kwargs["counter_norm"]
    max_actions = kwargs["max_actions"]
    reward_mode = kwargs["reward_mode"]
    swap_ready_bonus = kwargs["swap_ready_bonus"]
    debug = kwargs["debug"]
    diag_mode = kwargs["diag_mode"]
    eval_epsilon = kwargs["eval_epsilon"]
    trace_max_episodes = kwargs["trace_max_episodes"]
    trace_failed_only = kwargs["trace_failed_only"]
    mask_null_action = kwargs["mask_null_action"]
    block_refresh_actions = kwargs["block_refresh_actions"]
    prefer_swap_when_ready = kwargs["prefer_swap_when_ready"]

    eval_repeater_chain = RepeaterChain(n, pgen, pswap, torch_device)

    eval_actions_taken = np.zeros(eval_episodes)
    eval_reward_per_episode = np.zeros(eval_episodes)
    eval_steps = np.zeros(eval_episodes)
    eval_final_state = np.zeros(eval_episodes, dtype=bool)
    eval_bad_state = np.zeros(eval_episodes, dtype=bool)
    eval_truncated = np.zeros(eval_episodes, dtype=bool)

    eval_ent_count = np.zeros(eval_episodes)
    eval_swap_count = np.zeros(eval_episodes)

    actions_dir = os.path.join(os.path.dirname(__file__), "..", "qamel", "outputs", "logs", "actions")
    actions_path = os.path.join(actions_dir, f"{n}_nodes.npy")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(
            f"Missing actions file at {actions_path}. Train first to ensure action ordering matches."
        )
    all_actions = np.load(actions_path)
    all_actions = torch.Tensor(all_actions).to(torch_device)
    print(f"Loaded {all_actions.size(0)} actions from {actions_path}")
    swap_action_count = sum(1 for i in range(all_actions.size(0)) if _action_has_swap(all_actions[i]))
    if diag_mode:
        print(f"Actions with any swap bit: {swap_action_count}")
    print("Evaluation mode: no training code will run.")
    no_op_candidates = (all_actions.view(all_actions.size(0), -1).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    no_op_idx = int(no_op_candidates[0].item()) if len(no_op_candidates) > 0 else None

    q_table = None
    policy_net = None
    all_states = None
    if obs_mode in DQN_OBS_MODES:
        checkpoint = torch.load(model_path, map_location=torch_device, weights_only=False)
        input_shape = checkpoint.get("input_shape", (3, n, n))
        checkpoint_obs_mode = checkpoint.get("obs_mode")
        if checkpoint_obs_mode is not None and checkpoint_obs_mode != obs_mode:
            raise ValueError(
                f"Checkpoint obs_mode='{checkpoint_obs_mode}' does not match requested obs_mode='{obs_mode}'."
            )
        num_actions = all_actions.size(0)
        # Legacy checkpoints carry no net_arch -> default "dqn" -> DQNNet (identical behaviour).
        net_arch = checkpoint.get("net_arch", "dqn")
        policy_net = build_dqn_net(input_shape, num_actions, net_arch).to(torch_device)
        try:
            policy_net.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint shape mismatch at {model_path}. Retrain with --force_train."
            ) from exc
        policy_net.eval()
        print(f"DQN checkpoint loaded from {model_path}")
    else:
        q_table = np.loadtxt(q_table_path)
        all_states = np.load(states_path)
        all_states = torch.Tensor(all_states).to(torch_device)
        if torch.is_tensor(q_table):
            q_table = q_table.cpu().numpy()

    start_time = time.time()
    trace_records = []
    ablation_empty_fallbacks = 0
    for eval_episode in track(range(eval_episodes), description=f"Evaluating for node {env_vars['n']}"):
        cumulative_reward = 0
        steps = 0
        done = False
        episode_recorded = False
        current_state = eval_repeater_chain.reset()
        selected_swap_actions = 0
        selected_diagonals = []
        episode_trace = {
            "episode_index": int(eval_episode),
            "steps": [],
        }

        while not done:
            obs_state = _select_observation(current_state, obs_mode, counter_norm)
            if obs_mode in ["counter_exposed", "counter_exposed_plus_ready"]:
                valid_indices = _get_valid_action_indices(current_state[0], all_actions)
                selected_valid_indices = _filter_eval_action_indices(
                    current_state[0],
                    all_actions,
                    valid_indices,
                    no_op_idx=no_op_idx,
                    mask_null_action=mask_null_action,
                    block_refresh_actions=block_refresh_actions,
                    prefer_swap_when_ready=prefer_swap_when_ready,
                )
                if len(selected_valid_indices) == 0:
                    selected_valid_indices = valid_indices
                    ablation_empty_fallbacks += 1
                if diag_mode and eval_episode == 0:
                    valid_with_swaps = sum(1 for i in selected_valid_indices if _action_has_swap(all_actions[i]))
                    print(
                        f"diag step={steps} valid_actions={len(selected_valid_indices)} valid_with_swaps={valid_with_swaps}"
                    )
                if len(selected_valid_indices) == 0:
                    action_id = no_op_idx if no_op_idx is not None else 0
                elif eval_epsilon > 0.0 and np.random.rand() < eval_epsilon:
                    action_id = int(np.random.choice(selected_valid_indices))
                else:
                    with torch.no_grad():
                        q_values = policy_net(obs_state.unsqueeze(0).to(torch_device)).squeeze(0)
                        valid_mask = torch.zeros(all_actions.size(0), dtype=torch.bool, device=q_values.device)
                        valid_mask[selected_valid_indices] = True
                        q_values[~valid_mask] = -1e9
                        action_id = int(torch.argmax(q_values).item())
            else:
                action_id = np.argmax(q_table[get_state_id(all_states, obs_state)])

            if diag_mode and eval_episode == 0:
                action_diag = torch.diagonal(all_actions[action_id], 0).cpu().tolist()
                if len(selected_diagonals) < 10:
                    selected_diagonals.append(action_diag)
                if _action_has_swap(all_actions[action_id]):
                    selected_swap_actions += 1
            pre_state_summary = _summarize_state(current_state)
            action_summary = _summarize_action(all_actions[action_id], state0=current_state[0], action_id=action_id, null_action_idx=no_op_idx)
            new_state = eval_repeater_chain.step(current_state, all_actions[action_id])
            steps += 1
            episode_status = get_episode_status(new_state, steps, max_actions)
            post_state_summary = _summarize_state(new_state)

            reward = compute_reward(
                new_state,
                episode_status["final_state"],
                episode_status["bad_state"] or episode_status["truncated"],
                reward_mode=reward_mode,
                swap_ready_bonus=swap_ready_bonus,
                prev_state0=current_state[0],
                action_matrix=all_actions[action_id],
            )
            cumulative_reward += reward

            if trace_max_episodes > 0:
                episode_trace["steps"].append(
                    {
                        "step_index": int(steps),
                        "pre_state": pre_state_summary,
                        "action": action_summary,
                        "reward": float(reward),
                        "post_state": post_state_summary,
                        "status": {
                            "final_state": bool(episode_status["final_state"]),
                            "bad_state": bool(episode_status["bad_state"]),
                            "truncated": bool(episode_status["truncated"]),
                            "done": bool(episode_status["done"]),
                        },
                    }
                )

            current_state = new_state 
            if episode_status["done"]:
                eval_reward_per_episode[eval_episode] = cumulative_reward
                eval_ent_count[eval_episode] = torch.amax(current_state[1]).item()
                eval_swap_count[eval_episode] = torch.amax(current_state[2]).item()
                eval_steps[eval_episode] = steps
                eval_final_state[eval_episode] = episode_status["final_state"]
                eval_bad_state[eval_episode] = episode_status["bad_state"]
                eval_truncated[eval_episode] = episode_status["truncated"]
                episode_recorded = True
                done = True

            if done and not episode_recorded:
                eval_reward_per_episode[eval_episode] = cumulative_reward
                eval_ent_count[eval_episode] = torch.amax(current_state[1]).item()
                eval_swap_count[eval_episode] = torch.amax(current_state[2]).item()
                eval_steps[eval_episode] = steps
                eval_final_state[eval_episode] = episode_status["final_state"]
                eval_bad_state[eval_episode] = episode_status["bad_state"]
                eval_truncated[eval_episode] = episode_status["truncated"]
                episode_recorded = True

        if trace_max_episodes > 0:
            episode_trace["outcome"] = {
                "final_state": bool(eval_final_state[eval_episode]),
                "bad_state": bool(eval_bad_state[eval_episode]),
                "truncated": bool(eval_truncated[eval_episode]),
                "total_return": float(eval_reward_per_episode[eval_episode]),
                "steps": int(eval_steps[eval_episode]),
                "ent_attempt_max": float(eval_ent_count[eval_episode]),
                "swap_attempt_max": float(eval_swap_count[eval_episode]),
            }
            should_store = (not trace_failed_only) or bool(eval_truncated[eval_episode] or eval_bad_state[eval_episode] or not eval_final_state[eval_episode])
            if should_store and len(trace_records) < trace_max_episodes:
                trace_records.append(episode_trace)

        if debug and (eval_episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"debug episode={eval_episode} steps={steps} final={episode_status['final_state']} "
                f"bad={episode_status['bad_state']} truncated={episode_status['truncated']} elapsed={elapsed:.2f}s"
            )
        if diag_mode and eval_episode == 0:
            print(f"Selected swap actions in episode 0: {selected_swap_actions}")
            print("First 10 selected action diagonals:")
            for diag in selected_diagonals:
                print(diag)

        
    return (
        eval_actions_taken,
        eval_reward_per_episode,
        eval_ent_count,
        eval_swap_count,
        eval_steps,
        eval_final_state,
        eval_bad_state,
        eval_truncated,
        trace_records,
        ablation_empty_fallbacks,
    )


def _stats_summary(values):
    return float(np.mean(values)), float(np.std(values))

def run_evaluation(n, pgen, pswap, eval_episodes, model_tag, obs_mode, states_path, q_table_path, model_path, counter_norm, max_actions, reward_mode, swap_ready_bonus, debug, diag_mode, eval_epsilon, seed, trace_max_episodes, trace_failed_only, trace_output_path, mask_null_action, block_refresh_actions, prefer_swap_when_ready):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_global_seed(seed, torch_device)

    env_vars = {
        "n": n,
        "pgen": pgen,
        "pswap": pswap,
        "eval_episodes": eval_episodes
    }

    (
        _,
        rewards,
        ent_counts,
        swap_counts,
        steps,
        final_states,
        bad_states,
        truncated_states,
        trace_records,
        ablation_empty_fallbacks,
    ) = evaluate_q_table(
        env_vars,
        eval_episodes=eval_episodes,
        torch_device=torch_device,
        obs_mode=obs_mode,
        states_path=states_path,
        q_table_path=q_table_path,
        model_path=model_path,
        counter_norm=counter_norm,
        max_actions=max_actions,
        reward_mode=reward_mode,
        swap_ready_bonus=swap_ready_bonus,
        debug=debug,
        diag_mode=diag_mode,
        eval_epsilon=eval_epsilon,
        trace_max_episodes=trace_max_episodes,
        trace_failed_only=trace_failed_only,
        mask_null_action=mask_null_action,
        block_refresh_actions=block_refresh_actions,
        prefer_swap_when_ready=prefer_swap_when_ready,
    )

    output_paths = _build_evaluation_artifact_paths(
        n,
        pgen,
        pswap,
        model_tag,
        eval_episodes,
        max_actions,
        eval_epsilon,
        seed,
    )

    # Save results
    os.makedirs("qamel/outputs/results/ent_counts", exist_ok=True)
    os.makedirs("qamel/outputs/results/swap_counts", exist_ok=True)
    np.savetxt(f"qamel/outputs/results/ent_counts/{n}_nodes.txt", ent_counts)
    np.savetxt(f"qamel/outputs/results/swap_counts/{n}_nodes.txt", swap_counts)

    os.makedirs("qamel/outputs/results", exist_ok=True)
    csv_path = f"qamel/outputs/results/eval_n{n}_pgen{pgen}_pswap{pswap}_{model_tag}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "episode",
            "final_state",
            "bad_state",
            "truncated",
            "steps",
            "total_return",
            "ent_attempt_max",
            "swap_attempt_max",
        ])
        for episode_idx in range(eval_episodes):
            writer.writerow([
                episode_idx,
                bool(final_states[episode_idx]),
                bool(bad_states[episode_idx]),
                bool(truncated_states[episode_idx]),
                int(steps[episode_idx]),
                float(rewards[episode_idx]),
                float(ent_counts[episode_idx]),
                float(swap_counts[episode_idx]),
            ])

    success_mask = np.logical_and(final_states, np.logical_not(bad_states))
    success_rate = float(np.mean(success_mask))
    timeout_rate = float(np.mean(truncated_states))
    steps_mean, steps_std = _stats_summary(steps)
    ent_mean, ent_std = _stats_summary(ent_counts)
    swap_mean, swap_std = _stats_summary(swap_counts)
    rewards_mean, rewards_std = _stats_summary(rewards)

    os.makedirs(output_paths["evaluations_dir"], exist_ok=True)
    np.savetxt(output_paths["ent_counts_path"], ent_counts)
    np.savetxt(output_paths["swap_counts_path"], swap_counts)
    with open(output_paths["csv_path"], "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "episode",
            "final_state",
            "bad_state",
            "truncated",
            "steps",
            "total_return",
            "ent_attempt_max",
            "swap_attempt_max",
        ])
        for episode_idx in range(eval_episodes):
            writer.writerow([
                episode_idx,
                bool(final_states[episode_idx]),
                bool(bad_states[episode_idx]),
                bool(truncated_states[episode_idx]),
                int(steps[episode_idx]),
                float(rewards[episode_idx]),
                float(ent_counts[episode_idx]),
                float(swap_counts[episode_idx]),
            ])
    summary_payload = {
        "run_name": output_paths["run_name"],
        "model_tag": model_tag,
        "n": n,
        "pgen": pgen,
        "pswap": pswap,
        "obs_mode": obs_mode,
        "model_path": model_path,
        "eval_episodes": eval_episodes,
        "max_actions": max_actions,
        "eval_epsilon": eval_epsilon,
        "seed": seed,
        "reward_mode": reward_mode,
        "swap_ready_bonus": swap_ready_bonus,
        "mask_null_action": bool(mask_null_action),
        "block_refresh_actions": bool(block_refresh_actions),
        "prefer_swap_when_ready": bool(prefer_swap_when_ready),
        "ablation_empty_fallbacks": int(ablation_empty_fallbacks),
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "ent_mean": ent_mean,
        "ent_std": ent_std,
        "swap_mean": swap_mean,
        "swap_std": swap_std,
        "total_return_mean": rewards_mean,
        "total_return_std": rewards_std,
        "csv_path": output_paths["csv_path"],
        "ent_counts_path": output_paths["ent_counts_path"],
        "swap_counts_path": output_paths["swap_counts_path"],
    }
    with open(output_paths["summary_path"], "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2, sort_keys=True)
        summary_file.write("\n")

    if trace_output_path:
        os.makedirs(os.path.dirname(trace_output_path), exist_ok=True)
        trace_payload = {
            "model_tag": model_tag,
            "model_path": model_path,
            "obs_mode": obs_mode,
            "eval_epsilon": eval_epsilon,
            "seed": seed,
            "trace_failed_only": trace_failed_only,
            "trace_max_episodes": trace_max_episodes,
            "mask_null_action": bool(mask_null_action),
            "block_refresh_actions": bool(block_refresh_actions),
            "prefer_swap_when_ready": bool(prefer_swap_when_ready),
            "episodes": trace_records,
        }
        with open(trace_output_path, "w", encoding="utf-8") as trace_file:
            json.dump(trace_payload, trace_file, indent=2, sort_keys=True)
            trace_file.write("\n")
        print(f"Trace output: {trace_output_path}")

    print(f"Success rate: {success_rate:.4f}")
    print(f"Timeout rate: {timeout_rate:.4f}")
    print(f"Steps: {steps_mean:.4f} ± {steps_std:.4f}")
    print(f"Ent attempts: {ent_mean:.4f} ± {ent_std:.4f}")
    print(f"Swap attempts: {swap_mean:.4f} ± {swap_std:.4f}")
    print(f"Total return: {rewards_mean:.4f} ± {rewards_std:.4f}")
    print(f"Run evaluation CSV: {output_paths['csv_path']}")
    print(f"Run evaluation summary: {output_paths['summary_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--pgen", type=float, required=True)
    parser.add_argument("--pswap", type=float, required=True)
    parser.add_argument("--eval_episodes", type=int, required=True)
    parser.add_argument("--model_tag", type=str, default="baseline")
    parser.add_argument("--obs_mode", type=str, choices=["baseline", "counter_exposed", "counter_exposed_plus_ready"], default=None)
    parser.add_argument("--states_path", type=str, default=None)
    parser.add_argument("--q_table_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--counter_norm", type=float, default=20.0)
    parser.add_argument("--max_actions", type=int, default=100)
    parser.add_argument("--reward_mode", type=str, choices=sorted(SUPPORTED_REWARD_MODES), default="base")
    parser.add_argument("--swap_ready_bonus", type=float, default=0.5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--diag", action="store_true")
    parser.add_argument("--eval_epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trace_max_episodes", type=int, default=0)
    parser.add_argument("--trace_failed_only", action="store_true")
    parser.add_argument("--trace_output_path", type=str, default=None)
    parser.add_argument("--mask_null_action", action="store_true")
    parser.add_argument("--block_refresh_actions", action="store_true")
    parser.add_argument("--prefer_swap_when_ready", action="store_true")

    args = parser.parse_args()

    # Call function with parsed arguments
    obs_mode = _resolve_obs_mode(args.obs_mode, args.model_tag)

    if obs_mode in DQN_OBS_MODES:
        states_path = None
    else:
        states_path = args.states_path or f"qamel/outputs/logs/states/{args.n}_nodes.npy"

    q_table_path = None if obs_mode in DQN_OBS_MODES else (args.q_table_path or f"qamel/q_table_storage/{args.n}_nodes.txt")
    output_paths = _build_evaluation_artifact_paths(
        args.n,
        args.pgen,
        args.pswap,
        args.model_tag,
        args.eval_episodes,
        args.max_actions,
        args.eval_epsilon,
        args.seed,
    )
    legacy_model_path = f"qamel/outputs/models/dqn_n{args.n}_pgen{args.pgen}_pswap{args.pswap}_{args.model_tag}.pt"
    default_model_path = output_paths["run_model_path"] if os.path.exists(output_paths["run_model_path"]) else legacy_model_path
    model_path = args.model_path or default_model_path

    run_evaluation(
        args.n,
        args.pgen,
        args.pswap,
        args.eval_episodes,
        args.model_tag,
        obs_mode,
        states_path,
        q_table_path,
        model_path,
        args.counter_norm,
        args.max_actions,
        args.reward_mode,
        args.swap_ready_bonus,
        args.debug,
        args.diag,
        args.eval_epsilon,
        args.seed,
        args.trace_max_episodes,
        args.trace_failed_only,
        args.trace_output_path,
        args.mask_null_action,
        args.block_refresh_actions,
        args.prefer_swap_when_ready,
    )
