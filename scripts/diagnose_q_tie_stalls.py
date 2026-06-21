"""Diagnose whether greedy DQN stalls skip ready swaps because of Q-value ties.

This is a read-only checkpoint evaluation. It does not train, update, or rewrite the
checkpoint. For each episode that ends by the step limit, it preserves a step-level trace
containing the greedy action Q-value, the best available swap Q-value, the best currently
ready-swap Q-value, and the corresponding gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.dqn import build_dqn_net, preprocess_obs
from qamel.environment import RepeaterChain
from qamel.utils import generate_all_valid_actions, get_episode_status
from scripts.head_to_head import (
    _action_has_swap,
    _action_swaps_ready_node,
    _episode_seed,
    _load_actions,
    _load_config,
    _ready_interior_nodes,
    _resolve_checkpoint,
    _resolve_run_dir,
    _valid_action_indices,
)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _action_description(action: torch.Tensor) -> dict[str, Any]:
    n = action.size(0)
    generation_edges = [
        [i, j]
        for i in range(n)
        for j in range(i + 1, n)
        if bool(action[i, j].item() > 0)
    ]
    swap_nodes = [node for node in range(1, n - 1) if bool(action[node, node].item() > 0)]
    return {
        "generation_edges": generation_edges,
        "swap_nodes": swap_nodes,
        "is_no_op": not generation_edges and not swap_nodes,
    }


def _best_index(indices: list[int], q_values: torch.Tensor) -> int | None:
    if not indices:
        return None
    index_tensor = torch.tensor(indices, dtype=torch.long, device=q_values.device)
    offset = int(torch.argmax(q_values[index_tensor]).item())
    return indices[offset]


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "std": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _threshold_fractions(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "le_1e-6": None,
            "le_1e-3": None,
            "le_0.01": None,
            "le_0.1": None,
            "le_1.0": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "le_1e-6": float(np.mean(array <= 1e-6)),
        "le_1e-3": float(np.mean(array <= 1e-3)),
        "le_0.01": float(np.mean(array <= 0.01)),
        "le_0.1": float(np.mean(array <= 0.1)),
        "le_1.0": float(np.mean(array <= 1.0)),
    }


def _classify_gaps(
    gaps: list[float],
    *,
    near_tie_gap: float,
    large_gap: float,
) -> dict[str, str]:
    """Return an explicitly thresholded interpretation, not an unqualified causal claim."""
    if not gaps:
        return {
            "classification": "NO_SKIPPED_READY_SWAPS",
            "interpretation": "No step had an available ready swap that greedy skipped.",
        }

    stats = _distribution(gaps)
    median = float(stats["median"])
    p90 = float(stats["p90"])
    if p90 <= near_tie_gap:
        return {
            "classification": "NEAR_TIES",
            "interpretation": (
                f"At least 90% of skipped-ready-swap gaps are <= {near_tie_gap:g} Q units. "
                "This supports a deterministic ready-swap tie-break/filter as a targeted fix."
            ),
        }
    if median > large_gap:
        return {
            "classification": "LARGE_GAPS",
            "interpretation": (
                f"The median skipped-ready-swap gap exceeds {large_gap:g} Q units. "
                "The network genuinely ranks other valid actions above ready swaps."
            ),
        }
    return {
        "classification": "MIXED_GAPS",
        "interpretation": (
            f"The gaps are neither uniformly near-ties (p90 <= {near_tie_gap:g}) nor "
            f"predominantly large (median > {large_gap:g}). Inspect the quantiles and trace."
        ),
    }


def _reference_outcomes(path: str | None, seed: int, episodes: int) -> dict[int, dict[str, Any]]:
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle).get("rows", [])
    return {
        int(row["episode_index"]): row
        for row in rows
        if row.get("policy") == "dqn_greedy"
        and int(row.get("seed")) == seed
        and int(row.get("episode_index")) < episodes
    }


def _load_model(
    run_name: str,
    device: torch.device,
) -> tuple[
    str,
    str,
    dict[str, Any],
    torch.Tensor,
    torch.nn.Module,
    dict[str, Any],
]:
    run_dir = _resolve_run_dir(run_name)
    config = _load_config(run_dir)
    checkpoint_path = _resolve_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    env_config = config.get("env", {})
    train_config = config.get("train", {})
    n = int(env_config.get("n", checkpoint.get("n")))
    pgen = float(env_config.get("pgen", checkpoint.get("pgen")))
    pswap = float(env_config.get("pswap", checkpoint.get("pswap")))
    obs_mode = checkpoint.get(
        "obs_mode",
        train_config.get("obs_mode", "counter_exposed_plus_ready"),
    )
    counter_norm = float(
        checkpoint.get(
            "counter_norm",
            config.get("dqn_hyperparameters", {}).get("counter_norm", 20.0),
        )
    )
    net_arch = checkpoint.get("net_arch", train_config.get("net_arch", "dqn"))
    input_shape = checkpoint.get(
        "input_shape",
        (4 if obs_mode == "counter_exposed_plus_ready" else 3, n, n),
    )

    all_actions = _load_actions(n, device)
    policy_net = build_dqn_net(input_shape, all_actions.size(0), net_arch).to(device)
    policy_net.load_state_dict(checkpoint["model_state"])
    policy_net.eval()

    metadata = {
        "n": n,
        "pgen": pgen,
        "pswap": pswap,
        "obs_mode": obs_mode,
        "counter_norm": counter_norm,
        "net_arch": net_arch,
        "input_shape": list(input_shape),
        "num_actions": int(all_actions.size(0)),
    }
    return run_dir, checkpoint_path, config, all_actions, policy_net, metadata


def _step_q_trace(
    current_state: torch.Tensor,
    all_actions: torch.Tensor,
    policy_net: torch.nn.Module,
    obs_mode: str,
    counter_norm: float,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    valid_indices = _valid_action_indices(current_state[0], all_actions)
    if not valid_indices:
        raise RuntimeError("No valid action exists; expected at least the no-op action")

    obs = preprocess_obs(current_state, obs_mode, counter_norm).to(device)
    with torch.no_grad():
        q_values = policy_net(obs.unsqueeze(0)).squeeze(0)

    valid_mask = torch.zeros(all_actions.size(0), dtype=torch.bool, device=q_values.device)
    valid_mask[valid_indices] = True
    masked_q = q_values.clone()
    masked_q[~valid_mask] = -1e9
    chosen_idx = int(torch.argmax(masked_q).item())

    ready_nodes = _ready_interior_nodes(current_state[0])
    any_swap_indices = [idx for idx in valid_indices if _action_has_swap(all_actions[idx])]
    ready_swap_indices = [
        idx
        for idx in valid_indices
        if ready_nodes and _action_swaps_ready_node(all_actions[idx], ready_nodes)
    ]
    best_swap_idx = _best_index(any_swap_indices, q_values)
    best_ready_swap_idx = _best_index(ready_swap_indices, q_values)

    q_chosen = float(q_values[chosen_idx].item())
    q_best_swap = float(q_values[best_swap_idx].item()) if best_swap_idx is not None else None
    q_best_ready_swap = (
        float(q_values[best_ready_swap_idx].item())
        if best_ready_swap_idx is not None
        else None
    )
    chosen_takes_ready_swap = chosen_idx in ready_swap_indices
    ready_swap_available = best_ready_swap_idx is not None
    ready_swap_skipped = ready_swap_available and not chosen_takes_ready_swap
    ready_gap = q_chosen - q_best_ready_swap if q_best_ready_swap is not None else None

    if ready_gap is not None and ready_gap < -1e-5:
        raise AssertionError(
            "Greedy chosen Q is below best ready-swap Q; action/Q indexing is inconsistent: "
            f"chosen={q_chosen}, best_ready_swap={q_best_ready_swap}"
        )

    scale = (
        max(1e-12, abs(q_chosen), abs(q_best_ready_swap))
        if q_best_ready_swap is not None
        else None
    )
    trace = {
        "valid_action_count": len(valid_indices),
        "ready_nodes": ready_nodes,
        "ready_swap_available": ready_swap_available,
        "ready_swap_action_count": len(ready_swap_indices),
        "ready_swap_skipped": ready_swap_skipped,
        "chosen_takes_ready_swap": chosen_takes_ready_swap,
        "chosen_action_index": chosen_idx,
        "chosen_action": _action_description(all_actions[chosen_idx]),
        "q_chosen": q_chosen,
        "best_swap_action_index": best_swap_idx,
        "best_swap_action": (
            _action_description(all_actions[best_swap_idx])
            if best_swap_idx is not None
            else None
        ),
        "q_best_swap": q_best_swap,
        "q_chosen_minus_q_best_swap": (
            q_chosen - q_best_swap if q_best_swap is not None else None
        ),
        "best_ready_swap_action_index": best_ready_swap_idx,
        "best_ready_swap_action": (
            _action_description(all_actions[best_ready_swap_idx])
            if best_ready_swap_idx is not None
            else None
        ),
        "q_best_ready_swap": q_best_ready_swap,
        "q_chosen_minus_q_best_ready_swap": ready_gap,
        "relative_ready_swap_gap": ready_gap / scale if ready_gap is not None else None,
    }
    return chosen_idx, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-actions", type=int, default=150)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default=None, help="default: q_tie_diagnostic_seed<seed>.json in run dir")
    parser.add_argument(
        "--trace-out",
        default=None,
        help="default: q_tie_truncated_steps_seed<seed>.jsonl in run dir",
    )
    parser.add_argument(
        "--reference-per-episode",
        default=None,
        help="default: sibling head_to_head_per_episode.json when it exists",
    )
    parser.add_argument(
        "--near-tie-gap",
        type=float,
        default=0.1,
        help="NEAR_TIES requires p90 absolute Q-gap <= this value",
    )
    parser.add_argument(
        "--large-gap",
        type=float,
        default=1.0,
        help="LARGE_GAPS requires median absolute Q-gap > this value",
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_actions <= 0:
        parser.error("--episodes and --max-actions must be positive")

    device = _resolve_device(args.device)
    run_dir, checkpoint_path, _, all_actions, policy_net, model = _load_model(
        args.run_name,
        device,
    )
    output_path = args.out or os.path.join(
        run_dir,
        f"q_tie_diagnostic_seed{args.seed}_ep{args.episodes}.json",
    )
    trace_path = args.trace_out or os.path.join(
        run_dir,
        f"q_tie_truncated_steps_seed{args.seed}_ep{args.episodes}.jsonl",
    )
    reference_path = args.reference_per_episode
    if reference_path is None:
        candidate = os.path.join(run_dir, "head_to_head_per_episode.json")
        reference_path = candidate if os.path.exists(candidate) else None
    reference = _reference_outcomes(reference_path, args.seed, args.episodes)

    env = RepeaterChain(model["n"], model["pgen"], model["pswap"], device)
    truncated_traces: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    reference_mismatches: list[dict[str, Any]] = []

    for episode_index in range(args.episodes):
        episode_rng_seed = _episode_seed(args.seed, episode_index)
        torch.manual_seed(episode_rng_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(episode_rng_seed)

        current_state = env.reset()
        episode_trace: list[dict[str, Any]] = []
        status: dict[str, Any] | None = None

        for step in range(1, args.max_actions + 1):
            action_idx, trace = _step_q_trace(
                current_state,
                all_actions,
                policy_net,
                model["obs_mode"],
                model["counter_norm"],
                device,
            )
            trace.update(
                {
                    "episode_index": episode_index,
                    "episode_rng_seed": episode_rng_seed,
                    "step": step,
                }
            )
            episode_trace.append(trace)

            next_state = env.step(current_state, all_actions[action_idx])
            if not isinstance(next_state, torch.Tensor):
                raise TypeError(
                    "Environment returned a non-tensor transition for a supposedly valid action: "
                    f"episode={episode_index}, step={step}, action={action_idx}, value={next_state!r}"
                )
            status = get_episode_status(next_state, step, args.max_actions)
            current_state = next_state
            if status["done"]:
                break

        if status is None:
            raise AssertionError("Episode produced no transition")

        outcome = {
            "episode_index": episode_index,
            "steps": len(episode_trace),
            "success": bool(status["final_state"]),
            "bad_state": bool(status["bad_state"]),
            "truncated": bool(status["truncated"]),
        }
        outcomes.append(outcome)
        if status["truncated"]:
            truncated_traces.extend(episode_trace)

        reference_row = reference.get(episode_index)
        if reference_row is not None:
            reference_outcome = {
                "steps": int(reference_row["steps"]),
                "success": bool(reference_row["success"]),
                "truncated": bool(
                    not reference_row["success"]
                    and int(reference_row["steps"]) >= args.max_actions
                ),
            }
            observed = {
                "steps": outcome["steps"],
                "success": outcome["success"],
                "truncated": outcome["truncated"],
            }
            if reference_outcome != observed:
                reference_mismatches.append(
                    {
                        "episode_index": episode_index,
                        "reference": reference_outcome,
                        "observed": observed,
                    }
                )

    skipped_rows = [row for row in truncated_traces if row["ready_swap_skipped"]]
    gaps = [float(row["q_chosen_minus_q_best_ready_swap"]) for row in skipped_rows]
    relative_gaps = [float(row["relative_ready_swap_gap"]) for row in skipped_rows]
    ready_available_steps = sum(
        1 for row in truncated_traces if row["ready_swap_available"]
    )
    total_truncated_steps = len(truncated_traces)
    skipped_steps = len(skipped_rows)
    classification = _classify_gaps(
        gaps,
        near_tie_gap=args.near_tie_gap,
        large_gap=args.large_gap,
    )

    report = {
        "diagnostic": "greedy_ready_swap_q_gap_on_truncated_episodes",
        "read_only": True,
        "config": {
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "seed": args.seed,
            "episodes": args.episodes,
            "max_actions": args.max_actions,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            **model,
        },
        "episode_counts": {
            "evaluated": len(outcomes),
            "successes": sum(1 for row in outcomes if row["success"]),
            "bad_states": sum(1 for row in outcomes if row["bad_state"]),
            "truncated": sum(1 for row in outcomes if row["truncated"]),
        },
        "reference_replay_check": {
            "path": reference_path,
            "reference_rows_found": len(reference),
            "matching": len(reference) - len(reference_mismatches),
            "mismatch_count": len(reference_mismatches),
            "mismatches": reference_mismatches[:20],
            "note": (
                "A mismatch can indicate device-dependent near-tie action selection; "
                "prefer the same GPU class as the original evaluation."
            ),
        },
        "truncated_step_counts": {
            "total": total_truncated_steps,
            "ready_swap_available": ready_available_steps,
            "ready_swap_skipped": skipped_steps,
            "fraction_all_truncated_steps_ready_swap_available_but_skipped": (
                skipped_steps / total_truncated_steps if total_truncated_steps else None
            ),
            "fraction_ready_available_steps_skipped": (
                skipped_steps / ready_available_steps if ready_available_steps else None
            ),
        },
        "skipped_ready_swap_q_gap": {
            "definition": "Q(greedy chosen) - Q(best valid action swapping a currently ready node)",
            "absolute_q_units": _distribution(gaps),
            "absolute_threshold_fractions": _threshold_fractions(gaps),
            "relative_to_max_abs_q": _distribution(relative_gaps),
        },
        "verdict": {
            **classification,
            "near_tie_criterion": f"p90 absolute gap <= {args.near_tie_gap:g}",
            "large_gap_criterion": f"median absolute gap > {args.large_gap:g}",
            "causal_limit": (
                "Near-ties support an inference-time ready-swap tie-break/filter. They do not "
                "guarantee that prefer_swap_when_ready_train alone will improve the unfiltered "
                "greedy policy after retraining. Large gaps show ranking, but do not uniquely "
                "identify reward shaping as the only remedy."
            ),
        },
        "outputs": {"step_trace_jsonl": trace_path},
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as handle:
        for row in truncated_traces:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    gap_stats = report["skipped_ready_swap_q_gap"]["absolute_q_units"]
    step_counts = report["truncated_step_counts"]
    replay = report["reference_replay_check"]
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {report['config']['device_name']}")
    print(
        "episodes: "
        f"{report['episode_counts']['evaluated']} evaluated, "
        f"{report['episode_counts']['truncated']} truncated, "
        f"{report['episode_counts']['bad_states']} bad-state"
    )
    print(
        "reference replay: "
        f"{replay['matching']}/{replay['reference_rows_found']} match; "
        f"mismatches={replay['mismatch_count']}"
    )
    print(
        "truncated steps: "
        f"total={step_counts['total']} ready_available={step_counts['ready_swap_available']} "
        f"ready_skipped={step_counts['ready_swap_skipped']} "
        "fraction_all_steps_skipped="
        f"{step_counts['fraction_all_truncated_steps_ready_swap_available_but_skipped']}"
    )
    print(
        "skipped-ready Q gaps: "
        f"n={gap_stats['count']} median={gap_stats['median']} "
        f"p90={gap_stats['p90']} max={gap_stats['max']}"
    )
    print(f"VERDICT: {report['verdict']['classification']}")
    print(report["verdict"]["interpretation"])
    print(f"wrote: {output_path}")
    print(f"wrote: {trace_path}")


if __name__ == "__main__":
    main()
