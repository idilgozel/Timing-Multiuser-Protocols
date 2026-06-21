"""Re-score the pre-registered gate with a DIFFERENT gated policy, by re-aggregating an
existing head_to_head_per_episode.json. NO retraining, NO checkpoints, NO GPU.

Why this is admissible as "identical protocol, change only the policy": the per-episode
file already holds, for every (seed, episode_index), the success/steps of ALL three policies
(dqn_greedy, dqn_swapprefer, heuristic) under the same per-episode Common-Random-Numbers
seeding. Re-aggregating it with a different DQN policy reuses the exact same episodes; it does
not re-run anything.

The gate math below is a VERBATIM port of head_to_head._compute_gate2 (and the legacy paired-SR
delta in head_to_head.main), with the single change that the gated DQN policy name is a
parameter instead of the literal "dqn_greedy". A built-in self-check recomputes the GREEDY gate
and compares it to the stored head_to_head.json gate2 block; if they match, the swap-prefer
numbers produced by the same code are trustworthy.

IMPORTANT (per METRIC.md lines 11/13): swap-prefer is "reference only ... circular". This script
exists to QUANTIFY the reference-only result, NOT to redefine the headline gate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def _compute_gate(
    rows: list[dict[str, Any]],
    seeds: list[int],
    dqn_policy: str,
    gate1_delta: float = 0.02,
) -> dict[str, Any]:
    """Verbatim port of head_to_head._compute_gate2, parameterized on the gated DQN policy."""
    per_seed = []
    deltas: list[float] = []
    gate1_pass_count = 0
    for seed in seeds:
        dqn = {r["episode_index"]: r for r in rows if r["seed"] == seed and r["policy"] == dqn_policy}
        heur = {r["episode_index"]: r for r in rows if r["seed"] == seed and r["policy"] == "heuristic"}
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
            "SR_dqn": sr_dqn,
            "SR_heuristic": sr_heur,
            "gate1_pass": gate1,
            "mean_steps_dqn": ms_dqn,
            "mean_steps_heuristic": ms_heur,
            "delta": delta,
        })

    num = len(deltas)
    mean_delta = float(np.mean(deltas)) if deltas else None
    n_positive = sum(1 for d in deltas if d > 0)
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
            t_crit = 2.7764 if num == 5 else 2.0
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

    # Legacy paired-SR delta (head_to_head.main lines 508-527), gated policy parameterized.
    sr_deltas = [
        ps["SR_dqn"] - ps["SR_heuristic"]
        for ps in per_seed
        if ps["SR_dqn"] is not None and ps["SR_heuristic"] is not None
    ]
    sr_mean = float(np.mean(sr_deltas)) if sr_deltas else None
    sr_positive = sum(1 for d in sr_deltas if d > 0)
    mean_sr_dqn = float(np.mean([ps["SR_dqn"] for ps in per_seed if ps["SR_dqn"] is not None]))
    mean_sr_heur = float(np.mean([ps["SR_heuristic"] for ps in per_seed if ps["SR_heuristic"] is not None]))
    legacy_beat = bool(mean_sr_dqn > mean_sr_heur and sr_positive >= 4)

    return {
        "gated_policy": dqn_policy,
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
        "gate1_pass_count": gate1_pass_count,
        "gate1_required": required_gate1,
        "gate1_ok": gate1_ok,
        "gate2_sign_ok": gate2_sign_ok,
        "verdict": verdict,
        "verdict_drivers": drivers,
        "legacy_sr": {
            "mean_sr_dqn": mean_sr_dqn,
            "mean_sr_heuristic": mean_sr_heur,
            "mean_sr_delta": sr_mean,
            "seeds_positive": f"{sr_positive}/{len(seeds)}",
            "beat": legacy_beat,
        },
    }


def _truncation_diag(rows: list[dict[str, Any]], policy: str, seed: int, max_actions: int) -> dict[str, Any]:
    """Of `policy`'s FAILURES on one seed: steps==max_actions => truncation/step-limit;
    steps<max_actions => genuine dead-end (env reached a bad/terminal-bad state early).
    (The Q-value tie-loop question cannot be answered from this file; see report.)"""
    fails = [r for r in rows if r["seed"] == seed and r["policy"] == policy and not r["success"]]
    n_fail = len(fails)
    trunc = sum(1 for r in fails if r["steps"] >= max_actions)
    deadend = n_fail - trunc
    return {
        "policy": policy,
        "seed": seed,
        "max_actions": max_actions,
        "n_failures": n_fail,
        "truncations_steplimit": trunc,
        "dead_ends_early": deadend,
        "trunc_fraction_of_failures": (trunc / n_fail) if n_fail else None,
    }


def _fmt(v, spec=".4f"):
    return "n/a" if v is None else format(v, spec)


def _print_gate(tag: str, g: dict[str, Any]) -> None:
    print(f"\n### {tag} — gated policy = {g['gated_policy']}")
    print("| seed | |shared| | SR_dqn | SR_heur | Gate1 | steps_dqn | steps_heur | Delta |")
    print("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for r in g["per_seed"]:
        print(
            f"| {r['seed']} | {r['n_shared_solved']} | {_fmt(r['SR_dqn'], '.3f')} | "
            f"{_fmt(r['SR_heuristic'], '.3f')} | {'pass' if r['gate1_pass'] else 'FAIL'} | "
            f"{_fmt(r['mean_steps_dqn'], '.2f')} | {_fmt(r['mean_steps_heuristic'], '.2f')} | "
            f"{_fmt(r['delta'], '.3f')} |"
        )
    ci = g["ci95_mean_delta"]
    ls = g["legacy_sr"]
    print(f"\nlegacy SR: dqn={_fmt(ls['mean_sr_dqn'],'.4f')} heur={_fmt(ls['mean_sr_heuristic'],'.4f')} "
          f"delta={_fmt(ls['mean_sr_delta'],'.4f')} positive={ls['seeds_positive']} "
          f"-> {'BEAT' if ls['beat'] else 'NOT YET'}")
    print(f"mean Delta = {_fmt(g['mean_delta'],'.3f')} (min {_fmt(g['delta_min'],'.3f')}, max {_fmt(g['delta_max'],'.3f')})")
    print(f"paired t(df={g['df']}) = {_fmt(g['t_stat'],'.3f')}, 95% CI = [{_fmt(ci[0],'.3f')}, {_fmt(ci[1],'.3f')}]")
    print(f"Gate 1: {g['gate1_pass_count']}/{len(g['per_seed'])} pass (need {g['gate1_required']}); "
          f"sign {g['n_positive_of_total']} positive")
    print(f"VERDICT: {g['verdict']}  ({'; '.join(g['verdict_drivers'])})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-episode", required=True, help="path to head_to_head_per_episode.json")
    ap.add_argument("--gated-policy", default="dqn_swapprefer")
    ap.add_argument("--out", default=None, help="default: head_to_head_swapprefer.json next to input")
    ap.add_argument("--reference-h2h", default=None,
                    help="head_to_head.json for the greedy self-check (default: sibling)")
    ap.add_argument("--diag-policy", default="dqn_greedy")
    ap.add_argument("--diag-seed", type=int, default=None, help="default: first seed present")
    args = ap.parse_args()

    with open(args.per_episode, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = blob["rows"]
    cfg = blob.get("config", {})
    max_actions = int(cfg.get("max_actions", 100))
    seeds = sorted({int(r["seed"]) for r in rows})

    greedy = _compute_gate(rows, seeds, "dqn_greedy")
    gated = _compute_gate(rows, seeds, args.gated_policy)

    # --- self-check: recomputed greedy gate must match the stored head_to_head.json ---
    ref_path = args.reference_h2h or os.path.join(os.path.dirname(args.per_episode), "head_to_head.json")
    selfcheck = {"reference": ref_path, "ok": None, "note": None}
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as fh:
            stored = json.load(fh).get("gate2", {})
        sm, gm = stored.get("mean_delta"), greedy["mean_delta"]
        match = (sm is None and gm is None) or (sm is not None and gm is not None and abs(sm - gm) < 1e-6)
        match = match and stored.get("verdict") == greedy["verdict"]
        selfcheck["ok"] = bool(match)
        selfcheck["note"] = (f"stored greedy mean_delta={sm} verdict={stored.get('verdict')}; "
                             f"recomputed mean_delta={gm} verdict={greedy['verdict']}")
    else:
        selfcheck["note"] = "no reference head_to_head.json found; skipping self-check"

    diag_seed = args.diag_seed if args.diag_seed is not None else (seeds[0] if seeds else None)
    diag = _truncation_diag(rows, args.diag_policy, diag_seed, max_actions) if diag_seed is not None else None

    out = args.out or os.path.join(os.path.dirname(args.per_episode), "head_to_head_swapprefer.json")
    payload = {
        "source_per_episode": args.per_episode,
        "config": cfg,
        "metric_note": "METRIC.md lines 11/13: gated policy MUST be dqn_greedy; dqn_swapprefer is "
                       "reference-only and circular. This file QUANTIFIES the reference-only result.",
        "self_check_greedy_matches_stored": selfcheck,
        "gate_greedy_recomputed": greedy,
        "gate_gated_policy": gated,
        "truncation_diagnostic": diag,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(f"per-episode source: {args.per_episode}")
    print(f"seeds: {seeds}   max_actions: {max_actions}")
    print(f"\nSELF-CHECK (recomputed greedy vs stored head_to_head.json): "
          f"{'OK' if selfcheck['ok'] else 'MISMATCH/SKIPPED'} -- {selfcheck['note']}")
    _print_gate("GREEDY (the mandated, official gate)", greedy)
    _print_gate(f"REFERENCE-ONLY ({args.gated_policy}, NOT admissible per METRIC.md)", gated)
    if diag:
        d = diag
        print(f"\n### Truncation diagnostic — {d['policy']} failures on seed {d['seed']} (max_actions={d['max_actions']})")
        print(f"failures={d['n_failures']}  truncation/step-limit={d['truncations_steplimit']}  "
              f"dead-end-early={d['dead_ends_early']}  "
              f"trunc_fraction_of_failures={_fmt(d['trunc_fraction_of_failures'],'.3f')}")
        print("NOTE: the Q-value tie-loop question needs an instrumented checkpoint re-run "
              "(Q-gaps are not stored in per_episode.json). See report.")
    print(f"\nwrote: {out}")


if __name__ == "__main__":
    main()
