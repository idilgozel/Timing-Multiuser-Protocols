"""Compare DQN and heuristic policies on a repeater-chain environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.environment import RepeaterChain
from qamel.utils import (
    check_if_bad_state,
    check_if_final_state,
    compute_reward,
    count_swap_ready_nodes,
    generate_all_valid_actions,
    get_episode_status,
    is_action_valid_given_state,
)
from qamel.dqn import build_dqn_net, preprocess_obs


POLICIES = ("dqn_greedy", "dqn_swapprefer", "heuristic")


def _set_global_seed(seed: int, torch_device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_EPISODE_SEED_PRIME = 1_000_003  # large prime: spreads (S, i) pairs without collisions in range


def _episode_seed(seed: int, episode_index: int) -> int:
    """Deterministic per-episode env seed derived from (top-level seed S, episode index i).

    Implements Common Random Numbers: episode i is reseeded to the SAME value for every
    policy, so all policies face the identical environment RNG stream for that episode.
    """
    return (seed * _EPISODE_SEED_PRIME + episode_index) % (2**31 - 1)


def _resolve_run_dir(run_name: str) -> str:
    if os.path.isdir(run_name):
        return run_name
    return os.path.join("qamel", "outputs", "runs", run_name)


def _resolve_checkpoint(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "checkpoints", "best_eval.pt"),
        os.path.join(run_dir, "best_eval.pt"),
        os.path.join(run_dir, "model.pt"),
    ]
    checkpoint_path = next((path for path in candidates if os.path.exists(path)), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found. Looked for: {candidates}")
    return checkpoint_path


def _load_config(run_dir: str) -> dict[str, Any]:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing run config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_actions(n: int, torch_device: torch.device) -> torch.Tensor:
    actions_path = os.path.join("qamel", "outputs", "logs", "actions", f"{n}_nodes.npy")
    if os.path.exists(actions_path):
        actions = torch.tensor(np.load(actions_path), dtype=torch.float32)
    else:
        actions = generate_all_valid_actions(n)
    return actions.to(torch_device)


_VALID_INDEX_CACHE: dict[tuple[int, tuple[int, ...]], tuple[int, ...]] = {}


def _valid_action_indices(state0: torch.Tensor, all_actions: torch.Tensor) -> list[int]:
    key = (state0.size(0), tuple((state0 != 0).detach().flatten().to(torch.uint8).cpu().tolist()))
    cached = _VALID_INDEX_CACHE.get(key)
    if cached is not None:
        return list(cached)
    valid = [
        idx
        for idx in range(all_actions.size(0))
        if is_action_valid_given_state(state0, all_actions[idx])
    ]
    _VALID_INDEX_CACHE[key] = tuple(valid)
    return valid


def _ready_interior_nodes(state0: torch.Tensor) -> list[int]:
    if state0.size(0) <= 2:
        return []
    degrees = torch.count_nonzero(state0, dim=1)
    return [int(idx.item()) + 1 for idx in (degrees[1:-1] == 2).nonzero(as_tuple=False).flatten()]


def _action_has_generation(action_matrix: torch.Tensor) -> bool:
    return bool((torch.triu(action_matrix, diagonal=1) > 0).any().item())


def _action_has_swap(action_matrix: torch.Tensor) -> bool:
    diag = torch.diagonal(action_matrix, 0)
    if diag.numel() > 2:
        diag = diag[1:-1]
    return bool((diag > 0).any().item())


def _action_swaps_ready_node(action_matrix: torch.Tensor, ready_nodes: list[int]) -> bool:
    diag = torch.diagonal(action_matrix, 0)
    return any(bool(diag[node].item() > 0) for node in ready_nodes)


def _generation_only_action(action_matrix: torch.Tensor) -> bool:
    return _action_has_generation(action_matrix) and not _action_has_swap(action_matrix)


def _select_action(
    policy: str,
    current_state: torch.Tensor,
    all_actions: torch.Tensor,
    valid_indices: list[int],
    no_op_idx: int | None,
    policy_net: torch.nn.Module,
    obs_mode: str,
    counter_norm: float,
    eval_epsilon: float,
    torch_device: torch.device,
) -> int:
    ready_nodes = _ready_interior_nodes(current_state[0])

    # Touch the imported helper as a consistency guard: a positive count here means at
    # least one candidate swaps a currently-ready node under the same degree rule.
    if ready_nodes:
        ready_swap_actions = [
            idx for idx in valid_indices if count_swap_ready_nodes(current_state[0], all_actions[idx]) > 0
        ]
    else:
        ready_swap_actions = []

    if policy == "heuristic":
        if ready_nodes:
            candidates = [
                idx for idx in valid_indices if _action_swaps_ready_node(all_actions[idx], ready_nodes)
            ]
            if not candidates:
                candidates = ready_swap_actions or valid_indices
        else:
            candidates = [
                idx
                for idx in valid_indices
                if idx != no_op_idx and _generation_only_action(all_actions[idx])
            ]
            if not candidates:
                candidates = [idx for idx in valid_indices if idx != no_op_idx]
            if not candidates:
                candidates = valid_indices
        return random.choice(candidates)

    if policy == "dqn_swapprefer" and ready_nodes:
        candidates = [
            idx for idx in valid_indices if _action_swaps_ready_node(all_actions[idx], ready_nodes)
        ]
        if not candidates:
            candidates = valid_indices
    else:
        candidates = valid_indices

    if eval_epsilon > 0.0 and random.random() < eval_epsilon:
        return random.choice(candidates)

    obs = preprocess_obs(current_state, obs_mode, counter_norm).to(torch_device)
    with torch.no_grad():
        q_values = policy_net(obs.unsqueeze(0)).squeeze(0)
        valid_mask = torch.zeros(all_actions.size(0), dtype=torch.bool, device=q_values.device)
        valid_mask[candidates] = True
        q_values[~valid_mask] = -1e9
        return int(torch.argmax(q_values).item())


def _evaluate_policy(
    policy: str,
    *,
    seed: int,
    n: int,
    pgen: float,
    pswap: float,
    all_actions: torch.Tensor,
    no_op_idx: int | None,
    policy_net: torch.nn.Module,
    obs_mode: str,
    counter_norm: float,
    episodes: int,
    max_actions: int,
    eval_epsilon: float,
    torch_device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = RepeaterChain(n, pgen, pswap, torch_device)
    successes = 0
    truncations = 0
    success_steps: list[int] = []
    returns: list[float] = []
    episode_rows: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        # Per-episode Common Random Numbers: reseed the GLOBAL torch generator from
        # (seed, episode_index) before each episode. RepeaterChain.step() draws its
        # pgen/pswap Bernoullis exclusively via torch.rand(1) on this global generator
        # (environment.py L33, L50), so this makes episode `episode_index` present the
        # identical env RNG stream to every policy -> any outcome difference is due to the
        # policy's actions alone, satisfying METRIC.md's "same index = same instance".
        torch.manual_seed(_episode_seed(seed, episode_index))
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(_episode_seed(seed, episode_index))
        current_state = env.reset()
        cumulative_reward = 0.0
        steps = 0
        done = False
        episode_status: dict[str, Any] | None = None

        while not done:
            valid_indices = _valid_action_indices(current_state[0], all_actions)
            if not valid_indices:
                action_idx = no_op_idx if no_op_idx is not None else 0
            else:
                action_idx = _select_action(
                    policy,
                    current_state,
                    all_actions,
                    valid_indices,
                    no_op_idx,
                    policy_net,
                    obs_mode,
                    counter_norm,
                    eval_epsilon,
                    torch_device,
                )

            next_state = env.step(current_state, all_actions[action_idx])
            steps += 1
            episode_status = get_episode_status(next_state, steps, max_actions)
            final_state = check_if_final_state(next_state) and not check_if_bad_state(next_state)
            reward = compute_reward(
                next_state,
                final_state,
                episode_status["bad_state"] or episode_status["truncated"],
                reward_mode="base",
                swap_ready_bonus=0.0,
                prev_state0=current_state[0],
                action_matrix=all_actions[action_idx],
            )
            cumulative_reward += reward
            current_state = next_state
            done = episode_status["done"]

        assert episode_status is not None
        success_flag = bool(episode_status["final_state"])  # final-and-not-bad (see get_episode_status)
        returns.append(float(cumulative_reward))
        if success_flag:
            successes += 1
            success_steps.append(steps)
        if episode_status["truncated"]:
            truncations += 1
        episode_rows.append({
            "seed": seed,
            "policy": policy,
            "episode_index": episode_index,
            "success": success_flag,
            "steps": int(steps),
            "return": float(cumulative_reward),
        })

    metrics = {
        "policy": policy,
        "success_rate": successes / episodes,
        "mean_steps": float(np.mean(success_steps)) if success_steps else None,
        "mean_return": float(np.mean(returns)),
        "truncated_fraction": truncations / episodes,
        "n_success": successes,
    }
    return metrics, episode_rows


def _stats(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"mean": None, "iqr": None}
    return {
        "mean": float(np.mean(finite)),
        "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
    }


def _summary_for_policy(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    sr = _stats([row["success_rate"] for row in rows])
    steps = _stats([row["mean_steps"] for row in rows])
    returns = _stats([row["mean_return"] for row in rows])
    truncated = _stats([row["truncated_fraction"] for row in rows])
    return {
        "mean_sr": sr["mean"],
        "iqr_sr": sr["iqr"],
        "mean_steps": steps["mean"],
        "iqr_steps": steps["iqr"],
        "mean_return": returns["mean"],
        "iqr_return": returns["iqr"],
        "mean_truncated_fraction": truncated["mean"],
        "iqr_truncated_fraction": truncated["iqr"],
    }


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _compute_gate2(
    episode_rows: list[dict[str, Any]],
    seeds: list[int],
    gate1_delta: float = 0.02,
) -> dict[str, Any]:
    """Pre-registered Gate-2 (METRIC.md): paired, shared-solved steps comparison.

    Per seed, restrict to episodes solved by BOTH dqn_greedy (filter OFF) and heuristic
    (matched by episode_index = same instance under per-episode CRN), then
    Delta = mean_steps_heuristic - mean_steps_dqn_greedy. Seeds are the unit of analysis.
    """
    per_seed = []
    deltas: list[float] = []
    gate1_pass_count = 0
    for seed in seeds:
        dqn = {r["episode_index"]: r for r in episode_rows if r["seed"] == seed and r["policy"] == "dqn_greedy"}
        heur = {r["episode_index"]: r for r in episode_rows if r["seed"] == seed and r["policy"] == "heuristic"}
        n_eps = max(len(dqn), len(heur))
        sr_dqn = sum(1 for r in dqn.values() if r["success"]) / len(dqn) if dqn else None
        sr_heur = sum(1 for r in heur.values() if r["success"]) / len(heur) if heur else None
        gate1 = bool(sr_dqn is not None and sr_heur is not None and sr_dqn >= sr_heur - gate1_delta)
        gate1_pass_count += int(gate1)
        shared = [i for i in dqn if i in heur and dqn[i]["success"] and heur[i]["success"]]
        if shared:
            ms_dqn = float(np.mean([dqn[i]["steps"] for i in shared]))
            ms_heur = float(np.mean([heur[i]["steps"] for i in shared]))
            delta = ms_heur - ms_dqn
        else:
            ms_dqn = ms_heur = delta = None
        if delta is not None:
            deltas.append(delta)
        per_seed.append({
            "seed": seed,
            "n_shared_solved": len(shared),
            "n_episodes": n_eps,
            "SR_dqn_greedy": sr_dqn,
            "SR_heuristic": sr_heur,
            "gate1_pass": gate1,
            "mean_steps_dqn_greedy": ms_dqn,
            "mean_steps_heuristic": ms_heur,
            "delta": delta,
        })

    num = len(deltas)
    mean_delta = float(np.mean(deltas)) if deltas else None
    n_positive = sum(1 for d in deltas if d > 0)
    # Paired t-test of the per-seed Deltas vs 0, and a t-interval for the CI (df = num-1).
    t_stat = ci_low = ci_high = p_two_sided = None
    if num >= 2:
        try:
            from scipy import stats
            res = stats.ttest_1samp(np.asarray(deltas, dtype=float), 0.0)
            t_stat = float(res.statistic)
            p_two_sided = float(res.pvalue)
            t_crit = float(stats.t.ppf(0.975, df=num - 1))
        except Exception:
            arr = np.asarray(deltas, dtype=float)
            sem0 = float(np.std(arr, ddof=1) / np.sqrt(num))
            t_stat = float(mean_delta / sem0) if sem0 > 0 else float("inf")
            t_crit = 2.7764 if num == 5 else 2.0  # df=4 95% two-sided fallback
        sem = float(np.std(np.asarray(deltas, dtype=float), ddof=1) / np.sqrt(num))
        ci_low = float(mean_delta - t_crit * sem)
        ci_high = float(mean_delta + t_crit * sem)

    required_gate1 = 4 if len(seeds) == 5 else int(np.ceil(0.8 * len(seeds)))
    gate1_ok = gate1_pass_count >= required_gate1
    gate2_sign_ok = bool(mean_delta is not None and mean_delta > 0 and num == len(seeds) and n_positive == len(seeds))
    verdict = "WIN" if (gate1_ok and gate2_sign_ok) else "NOT-YET"
    drivers = []
    if not gate1_ok:
        drivers.append(f"Gate1 failed ({gate1_pass_count}/{len(seeds)} < {required_gate1})")
    if mean_delta is None or mean_delta <= 0:
        drivers.append("mean Delta not > 0")
    if num != len(seeds) or n_positive != len(seeds):
        drivers.append(f"not all seeds positive ({n_positive}/{len(seeds)})")
    if not drivers:
        drivers.append(f"Gate1 {gate1_pass_count}/{len(seeds)} and {n_positive}/{len(seeds)} positive Deltas with mean Delta>0")

    sign_test_note = (
        "5/5 in the winning direction => one-sided sign-test p = 0.031; per METRIC.md the "
        "directional consistency (not the t p-value) carries significance at n=5."
        if len(seeds) == 5 else
        f"{n_positive}/{len(seeds)} positive; sign-test significance per METRIC.md applies at n=5."
    )

    return {
        "metric_source": "METRIC.md",
        "gate1_delta": gate1_delta,
        "per_seed": per_seed,
        "deltas": deltas,
        "mean_delta": mean_delta,
        "delta_min": float(min(deltas)) if deltas else None,
        "delta_max": float(max(deltas)) if deltas else None,
        "t_stat": t_stat,
        "df": num - 1 if num >= 2 else None,
        "ci95_mean_delta": [ci_low, ci_high],
        "t_pvalue_two_sided": p_two_sided,
        "n_positive_of_total": f"{n_positive}/{len(seeds)}",
        "sign_test_note": sign_test_note,
        "gate1_pass_count": gate1_pass_count,
        "gate1_required": required_gate1,
        "gate1_ok": gate1_ok,
        "gate2_sign_ok": gate2_sign_ok,
        "verdict": verdict,
        "verdict_drivers": drivers,
        "decision_rule": "WIN iff Gate1 >= required-of-N within delta AND mean Delta>0 AND all N seeds Delta>0.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _resolve_run_dir(args.run_name)
    config = _load_config(run_dir)
    checkpoint_path = _resolve_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    env_config = config.get("env", {})
    train_config = config.get("train", {})
    n = int(env_config.get("n", checkpoint.get("n")))
    pgen = float(env_config.get("pgen", checkpoint.get("pgen")))
    pswap = float(env_config.get("pswap", checkpoint.get("pswap")))
    obs_mode = checkpoint.get("obs_mode", train_config.get("obs_mode", "counter_exposed_plus_ready"))
    counter_norm = float(checkpoint.get("counter_norm", config.get("dqn_hyperparameters", {}).get("counter_norm", 20.0)))
    net_arch = checkpoint.get("net_arch", train_config.get("net_arch", "dqn"))
    input_shape = checkpoint.get("input_shape", (4 if obs_mode == "counter_exposed_plus_ready" else 3, n, n))

    all_actions = _load_actions(n, torch_device)
    num_actions = all_actions.size(0)
    no_op_candidates = (all_actions.view(num_actions, -1).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    no_op_idx = int(no_op_candidates[0].item()) if no_op_candidates.numel() > 0 else None

    policy_net = build_dqn_net(input_shape, num_actions, net_arch).to(torch_device)
    policy_net.load_state_dict(checkpoint["model_state"])
    policy_net.eval()

    per_run = []
    per_episode: list[dict[str, Any]] = []
    for seed in args.seeds:
        for policy in POLICIES:
            # Seed python/numpy once per (policy, seed); the env's torch stream is then
            # reseeded per episode inside _evaluate_policy for Common Random Numbers.
            _set_global_seed(seed, torch_device)
            metrics, episode_rows = _evaluate_policy(
                policy,
                seed=seed,
                n=n,
                pgen=pgen,
                pswap=pswap,
                all_actions=all_actions,
                no_op_idx=no_op_idx,
                policy_net=policy_net,
                obs_mode=obs_mode,
                counter_norm=counter_norm,
                episodes=args.episodes,
                max_actions=args.max_actions,
                eval_epsilon=args.eval_epsilon,
                torch_device=torch_device,
            )
            metrics["seed"] = seed
            per_run.append(metrics)
            per_episode.extend(episode_rows)

    summary: dict[str, Any] = {}
    for policy in POLICIES:
        summary[policy] = _summary_for_policy([row for row in per_run if row["policy"] == policy])

    by_seed = {
        seed: {row["policy"]: row for row in per_run if row["seed"] == seed}
        for seed in args.seeds
    }
    deltas = [
        by_seed[seed]["dqn_greedy"]["success_rate"] - by_seed[seed]["heuristic"]["success_rate"]
        for seed in args.seeds
    ]
    positive = sum(1 for delta in deltas if delta > 0)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))
    mean_greedy = summary["dqn_greedy"]["mean_sr"]
    mean_heuristic = summary["heuristic"]["mean_sr"]
    beat = bool(
        mean_greedy is not None
        and mean_heuristic is not None
        and mean_greedy > mean_heuristic
        and positive >= 4
    )
    summary["paired_delta"] = {
        "mean": mean_delta,
        "std": std_delta,
        "seeds_positive_of_total": f"{positive}/{len(args.seeds)}",
    }

    gate2 = _compute_gate2(per_episode, list(args.seeds))

    payload = {
        "config": {
            "n": n,
            "pgen": pgen,
            "pswap": pswap,
            "episodes": args.episodes,
            "max_actions": args.max_actions,
            "eval_epsilon": args.eval_epsilon,
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "obs_mode": obs_mode,
            "net_arch": net_arch,
        },
        "per_run": per_run,
        "summary": summary,
        "verdict": "BEAT" if beat else "NOT YET",
        "gate2": gate2,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    # Per-episode rows (mandatory): joinable on (seed, episode_index) across policies.
    per_episode_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "head_to_head_per_episode.json")
    with open(per_episode_path, "w", encoding="utf-8") as handle:
        json.dump({"config": payload["config"], "rows": per_episode}, handle, indent=2)
        handle.write("\n")

    print("| Policy | mean success | success IQR | mean steps | mean return | truncated frac |")
    print("|---|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        row = summary[policy]
        print(
            f"| {policy} | {_fmt(row['mean_sr'])} | {_fmt(row['iqr_sr'])} | "
            f"{_fmt(row['mean_steps'], '.2f')} | {_fmt(row['mean_return'], '.2f')} | "
            f"{_fmt(row['mean_truncated_fraction'])} |"
        )
    print()
    print(f"paired_delta(SR, legacy) mean={mean_delta:.4f} std={std_delta:.4f} positive={positive}/{len(args.seeds)}")
    print(f"legacy verdict (SR-based): {'BEAT' if beat else 'NOT YET'}")
    print()
    print("### Gate 2 (METRIC.md: paired shared-solved steps-to-span)")
    print("| seed | |shared| | SR_dqn | SR_heur | Gate1 | steps_dqn | steps_heur | Delta |")
    print("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for r in gate2["per_seed"]:
        print(
            f"| {r['seed']} | {r['n_shared_solved']} | {_fmt(r['SR_dqn_greedy'], '.3f')} | "
            f"{_fmt(r['SR_heuristic'], '.3f')} | {'pass' if r['gate1_pass'] else 'FAIL'} | "
            f"{_fmt(r['mean_steps_dqn_greedy'], '.2f')} | {_fmt(r['mean_steps_heuristic'], '.2f')} | "
            f"{_fmt(r['delta'], '.3f')} |"
        )
    ci = gate2["ci95_mean_delta"]
    print()
    print(f"mean Delta = {_fmt(gate2['mean_delta'], '.3f')}  (min {_fmt(gate2['delta_min'], '.3f')}, max {_fmt(gate2['delta_max'], '.3f')})")
    print(f"paired t(df={gate2['df']}) = {_fmt(gate2['t_stat'], '.3f')}, 95% CI = [{_fmt(ci[0], '.3f')}, {_fmt(ci[1], '.3f')}]")
    print(f"sign consistency: {gate2['n_positive_of_total']} positive. {gate2['sign_test_note']}")
    print(f"Gate 1: {gate2['gate1_pass_count']}/{len(args.seeds)} pass (need {gate2['gate1_required']}).")
    print(f"GATE-2 VERDICT: {gate2['verdict']}  ({'; '.join(gate2['verdict_drivers'])})")
    print()
    print(f"wrote: {args.out}")
    print(f"wrote: {per_episode_path}")


if __name__ == "__main__":
    main()
