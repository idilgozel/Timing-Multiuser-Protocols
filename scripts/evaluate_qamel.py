import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.environment import RepeaterChain
from qamel.utils import check_if_bad_state, check_if_final_state, reward_shape, get_state_id, generate_all_valid_actions

import numpy as np
import torch
import argparse
from rich.progress import track
import csv

from qamel.dqn import DQNNet, preprocess_obs

def _select_observation(state, obs_mode, counter_norm=None):
    return preprocess_obs(state, obs_mode, counter_norm)

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

    eval_repeater_chain = RepeaterChain(n, pgen, pswap, torch_device)

    eval_actions_taken = np.zeros(eval_episodes)
    eval_reward_per_episode = np.zeros(eval_episodes)
    eval_steps = np.zeros(eval_episodes)
    eval_final_state = np.zeros(eval_episodes, dtype=bool)
    eval_bad_state = np.zeros(eval_episodes, dtype=bool)

    eval_ent_count = np.zeros(eval_episodes)
    eval_swap_count = np.zeros(eval_episodes)

    actions_dir = os.path.join(os.path.dirname(__file__), "..", "qamel", "outputs", "logs", "actions")
    actions_path = os.path.join(actions_dir, f"{n}_nodes.npy")
    if os.path.exists(actions_path):
        all_actions = np.load(actions_path)
        all_actions = torch.Tensor(all_actions).to(torch_device)
    else:
        all_actions = generate_all_valid_actions(n).to(torch_device)
        os.makedirs(actions_dir, exist_ok=True)
        np.save(actions_path, all_actions.cpu().numpy())

    q_table = None
    policy_net = None
    all_states = None
    if obs_mode == "counter_exposed":
        checkpoint = torch.load(model_path, map_location=torch_device)
        input_shape = checkpoint.get("input_shape", (3, n, n))
        num_actions = all_actions.size(0)
        policy_net = DQNNet(input_shape, num_actions).to(torch_device)
        policy_net.load_state_dict(checkpoint["model_state"])
        policy_net.eval()
    else:
        q_table = np.loadtxt(q_table_path)
        all_states = np.load(states_path)
        all_states = torch.Tensor(all_states).to(torch_device)
        if torch.is_tensor(q_table):
            q_table = q_table.cpu().numpy()

    for eval_episode in track(range(eval_episodes), description=f"Evaluating for node {env_vars['n']}"):
        cumulative_reward = 0
        steps = 0
        done = False
        current_state = eval_repeater_chain.reset()

        while not done:
            obs_state = _select_observation(current_state, obs_mode, counter_norm)
            if obs_mode == "counter_exposed":
                with torch.no_grad():
                    q_values = policy_net(obs_state.unsqueeze(0).to(torch_device))
                    action_id = int(torch.argmax(q_values, dim=1).item())
            else:
                action_id = np.argmax(q_table[get_state_id(all_states, obs_state)])
            new_state = eval_repeater_chain.step(current_state, all_actions[action_id])

            bad_state = check_if_bad_state(new_state)
            final_state = check_if_final_state(new_state)

            reward = reward_shape(new_state, final_state, bad_state)
            cumulative_reward += reward
            steps += 1

            current_state = new_state 

            if (check_if_final_state(current_state) or check_if_bad_state(current_state)):
                eval_reward_per_episode[eval_episode] = cumulative_reward
                eval_ent_count[eval_episode] = torch.amax(current_state[1]).item()
                eval_swap_count[eval_episode] = torch.amax(current_state[2]).item()
                eval_steps[eval_episode] = steps
                eval_final_state[eval_episode] = final_state
                eval_bad_state[eval_episode] = bad_state
                done = True

        
    return (
        eval_actions_taken,
        eval_reward_per_episode,
        eval_ent_count,
        eval_swap_count,
        eval_steps,
        eval_final_state,
        eval_bad_state,
    )


def _stats_summary(values):
    return float(np.mean(values)), float(np.std(values))

def run_evaluation(n, pgen, pswap, eval_episodes, model_tag, obs_mode, states_path, q_table_path, model_path, counter_norm):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ) = evaluate_q_table(
        env_vars,
        eval_episodes=eval_episodes,
        torch_device=torch_device,
        obs_mode=obs_mode,
        states_path=states_path,
        q_table_path=q_table_path,
        model_path=model_path,
        counter_norm=counter_norm,
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
                int(steps[episode_idx]),
                float(rewards[episode_idx]),
                float(ent_counts[episode_idx]),
                float(swap_counts[episode_idx]),
            ])

    success_mask = np.logical_and(final_states, np.logical_not(bad_states))
    success_rate = float(np.mean(success_mask))
    steps_mean, steps_std = _stats_summary(steps)
    ent_mean, ent_std = _stats_summary(ent_counts)
    swap_mean, swap_std = _stats_summary(swap_counts)
    rewards_mean, rewards_std = _stats_summary(rewards)

    print(f"Success rate: {success_rate:.4f}")
    print(f"Steps: {steps_mean:.4f} ± {steps_std:.4f}")
    print(f"Ent attempts: {ent_mean:.4f} ± {ent_std:.4f}")
    print(f"Swap attempts: {swap_mean:.4f} ± {swap_std:.4f}")
    print(f"Total return: {rewards_mean:.4f} ± {rewards_std:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--pgen", type=float, required=True)
    parser.add_argument("--pswap", type=float, required=True)
    parser.add_argument("--eval_episodes", type=int, required=True)
    parser.add_argument("--model_tag", type=str, default="baseline")
    parser.add_argument("--obs_mode", type=str, choices=["baseline", "counter_exposed"], default=None)
    parser.add_argument("--states_path", type=str, default=None)
    parser.add_argument("--q_table_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--counter_norm", type=float, default=20.0)

    args = parser.parse_args()

    # Call function with parsed arguments
    obs_mode = args.obs_mode if args.obs_mode is not None else args.model_tag
    if obs_mode not in ["baseline", "counter_exposed"]:
        obs_mode = "baseline"

    if obs_mode == "counter_exposed":
        states_path = None
    else:
        states_path = args.states_path or f"qamel/outputs/logs/states/{args.n}_nodes.npy"

    q_table_path = None if obs_mode == "counter_exposed" else (args.q_table_path or f"qamel/q_table_storage/{args.n}_nodes.txt")
    model_path = args.model_path or f"qamel/outputs/models/dqn_n{args.n}_pgen{args.pgen}_pswap{args.pswap}_{args.model_tag}.pt"

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
    )
