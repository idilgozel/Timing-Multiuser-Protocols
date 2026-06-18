"""Stage-0 8-seed escalation robustness figures.

Loads per-training-seed paired-evaluation results and builds four publication-grade
figures (vector PDF + 300-dpi PNG) into figures/. For each seed it prefers the run's
head_to_head.json; if that file is not present locally (e.g. not yet synced from the
cluster) it falls back to the embedded REFERENCE table extracted from the Myriad .out
logs. Either way, every parsed value is asserted against REFERENCE within 0.01.

Run:  python scripts/make_stage0_escalation_figures.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "qamel" / "outputs" / "runs"
FIGDIR = REPO / "figures"
FIGDIR.mkdir(exist_ok=True)

SEEDS = [202, 303, 404, 505, 606, 707, 808, 909]
EVAL_SEEDS = [12345, 23456, 34567, 45678, 56789]
HEUR_SR, HEUR_STEPS = 0.855, 38.10

# ---- reference values (from Myriad .out logs) — used to verify parsing or as fallback ----
REF_GREEDY_SR = {202: 0.9504, 303: 0.9410, 404: 0.9800, 505: 0.9770,
                 606: 0.9666, 707: 0.9410, 808: 0.9498, 909: 0.9768}
REF_GREEDY_STEPS = {202: 17.49, 303: 16.00, 404: 18.33, 505: 19.88,
                    606: 16.63, 707: 19.09, 808: 16.93, 909: 20.66}
REF_MEAN_DELTA = {202: 20.663, 303: 22.096, 404: 20.088, 505: 18.536,
                  606: 21.603, 707: 19.021, 808: 21.156, 909: 17.889}
REF_CI = {202: (19.844, 21.482), 303: (21.186, 23.006), 404: (19.495, 20.681),
          505: (18.212, 18.860), 606: (20.730, 22.476), 707: (18.125, 19.916),
          808: (20.313, 21.999), 909: (16.704, 19.074)}
REF_PER_EVAL_DELTA = {
    202: [20.729, 21.320, 19.604, 20.579, 21.083],
    303: [21.747, 22.593, 21.359, 21.663, 23.120],
    404: [19.562, 20.684, 19.681, 20.425, 20.087],
    505: [18.447, 18.941, 18.363, 18.292, 18.637],
    606: [21.433, 22.411, 20.617, 22.140, 21.414],
    707: [19.061, 19.704, 18.036, 18.598, 19.703],
    808: [20.410, 21.007, 20.667, 21.661, 22.035],
    909: [16.965, 19.407, 17.536, 17.358, 18.178],
}
POOLED_MEAN_DELTA_REF = float(np.mean([REF_MEAN_DELTA[s] for s in SEEDS]))  # ~20.13

# ---- shared style (matches notebooks/stage0_figures.ipynb) ----
OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
             "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
             "vermillion": "#D55E00", "purple": "#CC79A7"}


def _latex_renders():
    if shutil.which("latex") is None:
        return False
    try:
        import io
        with matplotlib.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title(r"$x^2$")
            fig.savefig(io.BytesIO(), format="png")
            plt.close(fig)
        return True
    except Exception:
        return False


# The committed stage0 figures were rendered with mathtext (usetex OFF); match them so the
# escalation panel is visually identical and Unicode (Δ, −) renders via the serif font.
USE_TEX = False
rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "savefig.transparent": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save_fig(fig, name):
    pdf, png = FIGDIR / f"{name}.pdf", FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# ---- data loading: prefer JSON, fall back to reference ----
def _run_dir(s):
    return RUNS / f"dqn_n5_pgen0.4_pswap0.7_stage0_fullstack_s{s}"


def load_seed(s):
    """Return (record, source). record has greedy_sr/greedy_steps, swap_sr/swap_steps
    (or None), mean_delta, ci(lo,hi), per_eval_delta(list len 5)."""
    h2h = _run_dir(s) / "head_to_head.json"
    if h2h.exists():
        d = json.loads(h2h.read_text())
        summ = d.get("summary", {})

        def policy(name, key_sr=("mean_sr", "success_rate"), key_st=("mean_steps",)):
            blk = summ.get(name, {})
            sr = next((blk[k] for k in key_sr if k in blk), None)
            st = next((blk[k] for k in key_st if k in blk), None)
            return sr, st

        g_sr, g_st = policy("dqn_greedy")
        sw_sr, sw_st = policy("dqn_swapprefer")
        g2 = d.get("gate2", {})
        by_seed = {int(r.get("seed")): float(r.get("delta")) for r in g2.get("per_seed", [])}
        per_eval = [by_seed[es] for es in EVAL_SEEDS] if set(EVAL_SEEDS) <= set(by_seed) else None
        mean_delta = g2.get("mean_delta")
        ci = g2.get("ci95_mean_delta")
        if per_eval is not None and mean_delta is not None and ci is not None and g_sr is not None:
            return ({"greedy_sr": float(g_sr), "greedy_steps": float(g_st),
                     "swap_sr": (None if sw_sr is None else float(sw_sr)),
                     "swap_steps": (None if sw_st is None else float(sw_st)),
                     "mean_delta": float(mean_delta), "ci": (float(ci[0]), float(ci[1])),
                     "per_eval_delta": [float(x) for x in per_eval]}, "json")
    # fallback
    return ({"greedy_sr": REF_GREEDY_SR[s], "greedy_steps": REF_GREEDY_STEPS[s],
             "swap_sr": None, "swap_steps": None,
             "mean_delta": REF_MEAN_DELTA[s], "ci": REF_CI[s],
             "per_eval_delta": list(REF_PER_EVAL_DELTA[s])}, "reference")


records, sources = {}, {}
for s in SEEDS:
    records[s], sources[s] = load_seed(s)

# ============================ VERIFICATION ============================
print("=" * 78)
print(f"Stage-0 escalation figures — {len(SEEDS)} training seeds; data source per seed:")
print("seed | source    | greedy_SR | greedy_steps | meanΔ  | 95% CI            | row-meanΔ")
TOL = 0.01
for s in SEEDS:
    r = records[s]
    row_mean = float(np.mean(r["per_eval_delta"]))
    print(f"{s:4d} | {sources[s]:9s} | {r['greedy_sr']:.4f}    | {r['greedy_steps']:6.2f}       "
          f"| {r['mean_delta']:.3f} | [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {row_mean:.3f}")
    # assert against reference within tolerance
    assert abs(r["greedy_sr"] - REF_GREEDY_SR[s]) <= TOL, f"greedy_sr seed {s}"
    assert abs(r["greedy_steps"] - REF_GREEDY_STEPS[s]) <= TOL, f"greedy_steps seed {s}"
    assert abs(r["mean_delta"] - REF_MEAN_DELTA[s]) <= TOL, f"mean_delta seed {s}"
    assert abs(r["ci"][0] - REF_CI[s][0]) <= TOL and abs(r["ci"][1] - REF_CI[s][1]) <= TOL, f"ci seed {s}"
    for a, b in zip(r["per_eval_delta"], REF_PER_EVAL_DELTA[s]):
        assert abs(a - b) <= TOL, f"per_eval_delta seed {s}"
    # internal consistency: per-eval row mean == seed mean
    assert abs(row_mean - r["mean_delta"]) <= TOL, f"row-mean vs mean_delta seed {s}"

greedy_sr = np.array([records[s]["greedy_sr"] for s in SEEDS])
greedy_steps = np.array([records[s]["greedy_steps"] for s in SEEDS])
mean_deltas = np.array([records[s]["mean_delta"] for s in SEEDS])
all_eval_deltas = np.array([records[s]["per_eval_delta"] for s in SEEDS])  # 8x5
pooled_mean = float(mean_deltas.mean())
have_swap = all(records[s]["swap_sr"] is not None for s in SEEDS)

assert (greedy_sr > HEUR_SR).all(), "a greedy success rate is <= heuristic"
assert (all_eval_deltas > 0).all(), "a per-eval-seed delta is <= 0"
assert abs(pooled_mean - POOLED_MEAN_DELTA_REF) <= TOL, "pooled mean mismatch"
print("-" * 78)
print(f"OK: all {len(SEEDS)} greedy SR > {HEUR_SR}; all {all_eval_deltas.size} per-eval deltas > 0; "
      f"pooled meanΔ = {pooled_mean:.3f}")
print(f"swap-prefer overlay available: {have_swap}")
print("=" * 78)

x = np.arange(len(SEEDS))
seed_labels = [str(s) for s in SEEDS]
written = []

# ============================ FIGURE 1 — success robustness ============================
fig, ax = plt.subplots(figsize=(7.4, 4.3))
ax.axhspan(HEUR_SR - 0.02, HEUR_SR, color=OKABE_ITO["vermillion"], alpha=0.08, zorder=0)
ax.axhline(HEUR_SR, ls="--", lw=1.4, color=OKABE_ITO["vermillion"], zorder=1)
if have_swap:
    sw = [records[s]["swap_sr"] for s in SEEDS]
    ax.scatter(x, sw, s=55, facecolors="none", edgecolors=OKABE_ITO["skyblue"], lw=1.6,
               zorder=2, label="DQN (swap-prefer)")
ax.scatter(x, greedy_sr, s=72, color=OKABE_ITO["blue"], zorder=3, label="DQN (greedy)")
ax.text(len(SEEDS) - 0.5, HEUR_SR + 0.004, f"heuristic = {HEUR_SR:.3f}", color=OKABE_ITO["vermillion"],
        fontsize=9.5, ha="right", va="bottom")
ax.annotate("8/8 seeds clear the heuristic", xy=(0.5, greedy_sr.min()),
            xytext=(0.4, 0.892), color=OKABE_ITO["blue"], fontsize=10,
            arrowprops=dict(arrowstyle="->", color=OKABE_ITO["blue"], lw=1.0))
ax.set_xticks(x); ax.set_xticklabels(seed_labels)
ax.set_xlabel("training seed"); ax.set_ylabel("success rate")
ax.set_ylim(0.832, 1.003); ax.set_xlim(-0.5, len(SEEDS) - 0.5)
ax.set_title("Stage-0 success rate is robust across 8 training seeds")
if have_swap:
    ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
written += list(save_fig(fig, "fig_stage0_esc_success"))

# ============================ FIGURE 2 — paired step-advantage w/ CIs ============================
fig, ax = plt.subplots(figsize=(7.4, 4.3))
yerr = np.array([[records[s]["mean_delta"] - records[s]["ci"][0] for s in SEEDS],
                 [records[s]["ci"][1] - records[s]["mean_delta"] for s in SEEDS]])
ax.axhline(0, color="0.15", lw=1.6, zorder=1)
ax.errorbar(x, mean_deltas, yerr=yerr, fmt="o", ms=7, color=OKABE_ITO["green"],
            ecolor=OKABE_ITO["green"], elinewidth=1.6, capsize=4, zorder=3, label="per-seed mean Δ ± 95% CI")
ax.axhline(pooled_mean, color=OKABE_ITO["green"], lw=2.0, ls="-", alpha=0.85, zorder=2,
           label=f"pooled mean = {pooled_mean:.2f}")
ax.annotate("all 8 seeds positive; pooled mean ≈ 20.1 steps faster",
            xy=(3.5, pooled_mean), xytext=(0.2, 6.5), color="0.25", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))
ax.set_xticks(x); ax.set_xticklabels(seed_labels)
ax.set_xlabel("training seed"); ax.set_ylabel("Δ steps  (heuristic − DQN, shared-solved)")
ax.set_ylim(0, 24); ax.set_xlim(-0.5, len(SEEDS) - 0.5)
ax.set_title("Paired step-advantage holds across seeds")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
written += list(save_fig(fig, "fig_stage0_esc_delta"))

# ============================ FIGURE 3 — success vs steps Pareto ============================
fig, ax = plt.subplots(figsize=(6.8, 5.0))
ax.scatter(greedy_steps, greedy_sr, s=78, color=OKABE_ITO["blue"], zorder=3, label="DQN (greedy)")
if have_swap:
    ax.scatter([records[s]["swap_steps"] for s in SEEDS], [records[s]["swap_sr"] for s in SEEDS],
               s=58, facecolors="none", edgecolors=OKABE_ITO["skyblue"], lw=1.6, zorder=2,
               label="DQN (swap-prefer)")
ax.scatter([HEUR_STEPS], [HEUR_SR], s=210, marker="D", color=OKABE_ITO["vermillion"], zorder=4,
           label="heuristic")
ax.set_xlim(13, 41)
ax.set_ylim(0.84, 0.995)
ax.annotate("heuristic", xy=(HEUR_STEPS, HEUR_SR), xytext=(HEUR_STEPS - 0.8, HEUR_SR + 0.013),
            color=OKABE_ITO["vermillion"], fontsize=10.5, ha="right", va="bottom")
ax.annotate("better", xy=(20.5, 0.972), xytext=(30.5, 0.916), color="0.3", fontsize=12, style="italic",
            ha="center", arrowprops=dict(arrowstyle="->", color="0.3", lw=1.6))
ax.set_xlabel("mean steps to completion"); ax.set_ylabel("success rate")
ax.set_title("DQN policies dominate the heuristic on both axes")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
written += list(save_fig(fig, "fig_stage0_esc_pareto"))

# ============================ FIGURE 4 — 8x5 delta heatmap ============================
fig, ax = plt.subplots(figsize=(6.8, 5.4))
im = ax.imshow(all_eval_deltas, cmap="viridis", vmin=16, vmax=23, aspect="auto")
ax.set_xticks(np.arange(len(EVAL_SEEDS))); ax.set_xticklabels([str(e) for e in EVAL_SEEDS])
ax.set_yticks(np.arange(len(SEEDS))); ax.set_yticklabels(seed_labels)
ax.set_xlabel("evaluation seed"); ax.set_ylabel("training seed")
ax.grid(False)
for i in range(len(SEEDS)):
    for j in range(len(EVAL_SEEDS)):
        v = all_eval_deltas[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                color=("white" if v < 20.0 else "black"))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Δ steps (heuristic − DQN)")
ax.set_title("All 40 paired deltas positive (8 training × 5 eval seeds)")
fig.tight_layout()
written += list(save_fig(fig, "fig_stage0_esc_delta_heatmap"))

# ============================ FILE VERIFICATION ============================
print("\nOutput files:")
ok = True
for p in written:
    sz = p.stat().st_size if p.exists() else 0
    flag = "OK" if (p.exists() and sz > 0) else "MISSING/EMPTY"
    if flag != "OK":
        ok = False
    print(f"  {flag:13s} {p.relative_to(REPO)}  ({sz} bytes)")
assert ok and len(written) == 8, "expected 8 non-empty output files"

print("\nSUMMARY: "
      f"{len(SEEDS)} seeds | greedy SR min/max = {greedy_sr.min():.3f}/{greedy_sr.max():.3f} | "
      f"mean Δ min/max = {mean_deltas.min():.2f}/{mean_deltas.max():.2f} | "
      f"pooled mean Δ = {pooled_mean:.2f} steps")
