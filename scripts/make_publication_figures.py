"""Build publication-style figures and analysis tables from real Qamel outputs.

The script intentionally reads only existing output files. It does not invent
missing results and does not run training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BUDGET_METRICS = [
    ("success_rate", "Success Rate", "success_rate_vs_budget"),
    ("mean_return", "Mean Return", "mean_return_vs_budget"),
    ("mean_steps", "Mean Steps", "mean_steps_vs_budget"),
    ("mean_ent_attempt_max", "Mean Generation-Attempt Max", "generation_attempts_vs_budget"),
    ("mean_swap_attempt_max", "Mean Swap-Attempt Max", "swap_attempts_vs_budget"),
    ("truncated_fraction", "Truncated Fraction", "truncated_fraction_vs_budget"),
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _binomial_se(success_rate: float | None, episodes: int | None) -> float | None:
    if success_rate is None or episodes is None or episodes <= 0:
        return None
    return math.sqrt(max(success_rate * (1.0 - success_rate), 0.0) / episodes)


def _seed_from_text(text: str) -> int | None:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else None


def _study_family(study: str) -> str:
    return re.sub(r"_seed\d+$", "", study)


def _normalise_policy(policy: str) -> str:
    policy = policy.replace("A_dqn_greedy", "dqn_greedy")
    policy = policy.replace("B_dqn_prefer_swap", "dqn_swapprefer")
    policy = policy.replace("C_heuristic", "heuristic")
    return policy


def collect_budget_rows(studies_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(studies_root.glob("*/budget_*/eval_summary.json")):
        budget_match = re.search(r"budget_(\d+)$", summary_path.parent.name)
        if not budget_match:
            continue
        study = summary_path.parent.parent.name
        payload = _read_json(summary_path)
        success_rate = _safe_float(payload.get("success_rate"))
        eval_episodes = payload.get("eval_episodes")
        eval_episodes = int(eval_episodes) if eval_episodes is not None else None
        row = {
            "source_path": str(summary_path),
            "study": study,
            "study_family": _study_family(study),
            "seed": payload.get("seed") if payload.get("seed") is not None else _seed_from_text(study),
            "budget": int(budget_match.group(1)),
            "n": payload.get("n"),
            "pgen": payload.get("pgen"),
            "pswap": payload.get("pswap"),
            "obs_mode": payload.get("obs_mode"),
            "eval_episodes": eval_episodes,
            "success_rate": success_rate,
            "success_rate_se_binomial": _binomial_se(success_rate, eval_episodes),
            "success_rate_ci95_halfwidth": (
                1.96 * _binomial_se(success_rate, eval_episodes)
                if _binomial_se(success_rate, eval_episodes) is not None
                else None
            ),
            "mean_return": _safe_float(payload.get("total_return_mean")),
            "mean_steps": _safe_float(payload.get("steps_mean")),
            "mean_ent_attempt_max": _safe_float(payload.get("ent_mean")),
            "mean_swap_attempt_max": _safe_float(payload.get("swap_mean")),
            "truncated_fraction": _safe_float(payload.get("timeout_rate")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def collect_eval_inventory(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.glob("*/evaluations/*_summary.json")):
        payload = _read_json(summary_path)
        success_rate = _safe_float(payload.get("success_rate"))
        eval_episodes = payload.get("eval_episodes")
        eval_episodes = int(eval_episodes) if eval_episodes is not None else None
        rows.append(
            {
                "source_path": str(summary_path),
                "run_name": payload.get("run_name") or summary_path.parents[1].name,
                "model_tag": payload.get("model_tag"),
                "n": payload.get("n"),
                "pgen": payload.get("pgen"),
                "pswap": payload.get("pswap"),
                "obs_mode": payload.get("obs_mode"),
                "seed": payload.get("seed"),
                "eval_episodes": eval_episodes,
                "max_actions": payload.get("max_actions"),
                "eval_epsilon": payload.get("eval_epsilon"),
                "mask_null_action": payload.get("mask_null_action"),
                "block_refresh_actions": payload.get("block_refresh_actions"),
                "prefer_swap_when_ready": payload.get("prefer_swap_when_ready"),
                "success_rate": success_rate,
                "success_rate_se_binomial": _binomial_se(success_rate, eval_episodes),
                "success_rate_ci95_halfwidth": (
                    1.96 * _binomial_se(success_rate, eval_episodes)
                    if _binomial_se(success_rate, eval_episodes) is not None
                    else None
                ),
                "timeout_rate": _safe_float(payload.get("timeout_rate")),
                "mean_steps": _safe_float(payload.get("steps_mean")),
                "mean_return": _safe_float(payload.get("total_return_mean")),
                "mean_ent_attempt_max": _safe_float(payload.get("ent_mean")),
                "mean_swap_attempt_max": _safe_float(payload.get("swap_mean")),
            }
        )
    return pd.DataFrame(rows)


def collect_ablation_rows(studies_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(studies_root.glob("*/diagnostics/eval_ablations/combined_ablation_summary.csv")):
        frame = pd.read_csv(csv_path)
        frame.insert(0, "study", csv_path.parents[2].name)
        frame.insert(1, "source_path", str(csv_path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_head_to_head_rows(head_to_head_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for json_path in sorted(head_to_head_root.glob("*.json")):
        payload = _read_json(json_path)
        source = json_path.stem
        run_name = payload.get("config", {}).get("run_dir") or payload.get("run_name")
        raw_rows = payload.get("per_run") or payload.get("rows") or []
        for row in raw_rows:
            success_rate = _safe_float(row.get("success_rate"))
            episodes = row.get("episodes") or payload.get("config", {}).get("episodes") or payload.get("episodes")
            episodes = int(episodes) if episodes is not None else None
            rows.append(
                {
                    "source": source,
                    "source_path": str(json_path),
                    "run_name": run_name,
                    "policy": _normalise_policy(str(row.get("policy"))),
                    "seed": row.get("seed"),
                    "episodes": episodes,
                    "success_rate": success_rate,
                    "success_rate_se_binomial": _binomial_se(success_rate, episodes),
                    "mean_steps": _safe_float(row.get("mean_steps")),
                    "mean_return": _safe_float(row.get("mean_return")),
                    "truncated_fraction": _safe_float(row.get("truncated_fraction")),
                    "n_success": row.get("n_success"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_budget_rows(budget_df: pd.DataFrame) -> pd.DataFrame:
    if budget_df.empty:
        return pd.DataFrame()

    grouped = []
    for keys, frame in budget_df.groupby(["study_family", "budget"], dropna=False):
        study_family, budget = keys
        row: dict[str, Any] = {
            "study_family": study_family,
            "budget": budget,
            "n_rows": len(frame),
            "n_seeds": frame["seed"].nunique(dropna=True) if "seed" in frame else None,
            "source_studies": ";".join(sorted(frame["study"].astype(str).unique())),
        }
        for metric, _, _ in BUDGET_METRICS:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.nan
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["study_family", "budget"])


def aggregate_head_to_head_rows(h2h_df: pd.DataFrame) -> pd.DataFrame:
    if h2h_df.empty:
        return pd.DataFrame()
    grouped = []
    for keys, frame in h2h_df.groupby(["source", "policy"], dropna=False):
        source, policy = keys
        row: dict[str, Any] = {"source": source, "policy": policy, "n_rows": len(frame)}
        for metric in ["success_rate", "mean_steps", "mean_return", "truncated_fraction"]:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_se"] = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.nan
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["source", "policy"])


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [pdf_path, png_path]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def plot_budget_metric(budget_df: pd.DataFrame, metric: str, ylabel: str, out_dir: Path, stem: str) -> list[Path]:
    plot_df = budget_df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return []

    collapsed = (
        plot_df.groupby(["study", "budget"], as_index=False)
        .agg({metric: "mean", "success_rate_se_binomial": "mean"})
        .sort_values(["study", "budget"])
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for study, frame in collapsed.groupby("study"):
        ax.plot(frame["budget"], frame[metric], marker="o", linewidth=1.5, label=study)
        if metric == "success_rate" and frame["success_rate_se_binomial"].notna().any():
            se = frame["success_rate_se_binomial"].fillna(0.0)
            ax.fill_between(frame["budget"], frame[metric] - se, frame[metric] + se, alpha=0.12)

    ax.set_xlabel("Training budget (episodes)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if metric in {"success_rate", "truncated_fraction"}:
        ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)


def plot_combined_budget_summary(budget_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if budget_df.empty:
        return []
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True)
    for ax, (metric, ylabel, _) in zip(axes.ravel(), BUDGET_METRICS):
        plot_df = budget_df.dropna(subset=[metric]).copy()
        for study, frame in plot_df.groupby("study"):
            frame = frame.sort_values("budget")
            ax.plot(frame["budget"], frame[metric], marker="o", linewidth=1.2, label=study)
        ax.set_title(ylabel)
        ax.set_xscale("log")
        if metric in {"success_rate", "truncated_fraction"}:
            ax.set_ylim(-0.03, 1.03)
    for ax in axes[-1, :]:
        ax.set_xlabel("Budget")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(fig, out_dir, "combined_budget_summary")


def plot_ablation_success(ablation_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    if ablation_df.empty or "success_rate" not in ablation_df:
        return []
    frame = ablation_df.dropna(subset=["success_rate"]).copy()
    if frame.empty:
        return []
    frame["label"] = frame["study"].astype(str) + " ckpt " + frame["checkpoint"].astype(str)
    labels = list(dict.fromkeys(frame["label"].tolist()))
    variants = list(dict.fromkeys(frame["variant"].astype(str).tolist()))
    width = 0.8 / max(len(variants), 1)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(labels)), 3.8))
    for i, variant in enumerate(variants):
        values = []
        for label in labels:
            match = frame[(frame["label"] == label) & (frame["variant"].astype(str) == variant)]
            values.append(float(match["success_rate"].mean()) if not match.empty else np.nan)
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, values, width=width, label=variant)
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_dir, "ablation_success_comparison")


def plot_head_to_head(h2h_agg: pd.DataFrame, out_dir: Path) -> list[Path]:
    if h2h_agg.empty:
        return []
    frame = h2h_agg.dropna(subset=["success_rate_mean"]).copy()
    if frame.empty:
        return []
    sources = list(dict.fromkeys(frame["source"].astype(str).tolist()))
    policies = list(dict.fromkeys(frame["policy"].astype(str).tolist()))
    width = 0.8 / max(len(policies), 1)
    x = np.arange(len(sources))

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(sources)), 3.8))
    for i, policy in enumerate(policies):
        values = []
        errors = []
        for source in sources:
            match = frame[(frame["source"].astype(str) == source) & (frame["policy"].astype(str) == policy)]
            values.append(float(match["success_rate_mean"].iloc[0]) if not match.empty else np.nan)
            errors.append(float(match["success_rate_se"].iloc[0]) if not match.empty and pd.notna(match["success_rate_se"].iloc[0]) else 0.0)
        ax.bar(
            x + (i - (len(policies) - 1) / 2) * width,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=policy,
        )
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=25, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, out_dir, "head_to_head_policy_comparison")


def write_manifest(paths: list[Path], tables_dir: Path) -> None:
    manifest = pd.DataFrame({"artifact": [str(path) for path in paths]})
    manifest.to_csv(tables_dir / "publication_artifacts_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies-root", type=Path, default=Path("qamel/outputs/studies"))
    parser.add_argument("--runs-root", type=Path, default=Path("qamel/outputs/runs"))
    parser.add_argument("--head-to-head-root", type=Path, default=Path("qamel/outputs/head_to_head"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/publication"))
    parser.add_argument("--tables-dir", type=Path, default=Path("qamel/outputs/analysis"))
    parser.add_argument(
        "--min-eval-episodes-for-figures",
        type=int,
        default=50,
        help="Minimum evaluation episodes required for a row to appear in figures. Raw tables keep all rows.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []

    budget_df = collect_budget_rows(args.studies_root)
    eval_df = collect_eval_inventory(args.runs_root)
    ablation_df = collect_ablation_rows(args.studies_root)
    h2h_df = collect_head_to_head_rows(args.head_to_head_root)
    budget_agg = aggregate_budget_rows(budget_df)
    h2h_agg = aggregate_head_to_head_rows(h2h_df)
    budget_plot_df = budget_df[
        pd.to_numeric(budget_df.get("eval_episodes"), errors="coerce").fillna(0) >= args.min_eval_episodes_for_figures
    ].copy()
    h2h_plot_df = h2h_df[
        pd.to_numeric(h2h_df.get("episodes"), errors="coerce").fillna(0) >= args.min_eval_episodes_for_figures
    ].copy()
    h2h_plot_agg = aggregate_head_to_head_rows(h2h_plot_df)

    table_outputs = {
        "budget_results.csv": budget_df,
        "budget_results_aggregate.csv": budget_agg,
        "budget_results_figure_subset.csv": budget_plot_df,
        "evaluation_inventory.csv": eval_df,
        "ablation_results.csv": ablation_df,
        "head_to_head_results.csv": h2h_df,
        "head_to_head_aggregate.csv": h2h_agg,
        "head_to_head_figure_subset.csv": h2h_plot_df,
    }
    for filename, frame in table_outputs.items():
        path = args.tables_dir / filename
        frame.to_csv(path, index=False)
        artifacts.append(path)

    for metric, ylabel, stem in BUDGET_METRICS:
        artifacts.extend(plot_budget_metric(budget_plot_df, metric, ylabel, args.figures_dir, stem))
    artifacts.extend(plot_combined_budget_summary(budget_plot_df, args.figures_dir))
    artifacts.extend(plot_ablation_success(ablation_df, args.figures_dir))
    artifacts.extend(plot_head_to_head(h2h_plot_agg, args.figures_dir))

    write_manifest(artifacts, args.tables_dir)
    artifacts.append(args.tables_dir / "publication_artifacts_manifest.csv")

    print("Generated artifacts:")
    for path in artifacts:
        print(path)


if __name__ == "__main__":
    main()
