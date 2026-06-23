"""Heuristic-only baseline at arbitrary n. NO checkpoint, NO GPU, NO neural network.

Reuses head_to_head._evaluate_policy with the 'heuristic' policy. That policy returns its action
before ever touching the network (head_to_head._select_action, the `policy == "heuristic"`
branch), so we pass policy_net=None safely. It uses the SAME per-episode Common-Random-Numbers
seeding as the gate, so the SR / mean-steps printed here are exactly the bar the DQN must clear
in the head_to_head 'heuristic' column. Device is forced to CPU.

Usage (login node, no GPU):
    python scripts/heuristic_baseline.py --n 7 --pgen 0.4 --pswap 0.7 \
        --max-actions 300 --episodes 1000 --seeds 12345 23456 34567 45678 56789
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# scripts/ is sys.path[0] when run as `python scripts/heuristic_baseline.py`, so this resolves.
from head_to_head import _evaluate_policy, _load_actions, _set_global_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--pgen", type=float, default=0.4)
    ap.add_argument("--pswap", type=float, default=0.7)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456, 34567, 45678, 56789])
    ap.add_argument("--max-actions", type=int, default=300)
    ap.add_argument("--out", default=None, help="optional path to write a JSON summary")
    args = ap.parse_args()

    device = torch.device("cpu")  # no GPU by design
    all_actions = _load_actions(args.n, device)
    num_actions = all_actions.size(0)
    no_op = (all_actions.view(num_actions, -1).sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    no_op_idx = int(no_op[0].item()) if no_op.numel() > 0 else None

    per_seed = []
    for seed in args.seeds:
        _set_global_seed(seed, device)
        metrics, _ = _evaluate_policy(
            "heuristic",
            seed=seed,
            n=args.n,
            pgen=args.pgen,
            pswap=args.pswap,
            all_actions=all_actions,
            no_op_idx=no_op_idx,
            policy_net=None,                       # heuristic never uses the net
            obs_mode="counter_exposed_plus_ready", # unused by heuristic
            counter_norm=20.0,                     # unused by heuristic
            episodes=args.episodes,
            max_actions=args.max_actions,
            eval_epsilon=0.0,
            torch_device=device,
        )
        metrics["seed"] = seed
        per_seed.append(metrics)

    srs = [m["success_rate"] for m in per_seed]
    steps = [m["mean_steps"] for m in per_seed if m["mean_steps"] is not None]
    truncs = [m["truncated_fraction"] for m in per_seed]
    agg = {
        "n": args.n,
        "pgen": args.pgen,
        "pswap": args.pswap,
        "max_actions": args.max_actions,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "mean_success_rate": float(np.mean(srs)),
        "mean_steps_over_successes": float(np.mean(steps)) if steps else None,
        "mean_truncated_fraction": float(np.mean(truncs)),
    }

    print(f"\nHEURISTIC baseline  n={args.n}  pgen={args.pgen}  pswap={args.pswap}  "
          f"max_actions={args.max_actions}  episodes={args.episodes}/seed")
    print("| seed | success_rate | mean_steps(success) | truncated_frac |")
    print("|---|---:|---:|---:|")
    for m in per_seed:
        ms = "n/a" if m["mean_steps"] is None else f"{m['mean_steps']:.2f}"
        print(f"| {m['seed']} | {m['success_rate']:.4f} | {ms} | {m['truncated_fraction']:.4f} |")
    msa = "n/a" if agg["mean_steps_over_successes"] is None else f"{agg['mean_steps_over_successes']:.2f}"
    print(f"\nBAR TO CLEAR -> mean SR = {agg['mean_success_rate']:.4f}   "
          f"mean steps = {msa}   mean truncated = {agg['mean_truncated_fraction']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"aggregate": agg, "per_seed": per_seed}, fh, indent=2)
            fh.write("\n")
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
