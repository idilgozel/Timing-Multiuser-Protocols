# Results Summary

Date: 2026-06-11

Scope: 1D repeater-chain Bell-pair scheduling in the current Qamel executable environment. No grid/fusion/GHZ claims are made from these runs.

## What Was Inspected

Canonical code paths:

- Training: `scripts/train_qamel.py`
- Evaluation: `scripts/evaluate_qamel.py`
- Head-to-head comparison: `scripts/head_to_head.py`
- Multi-budget study driver: `scripts/run_stage2_study.sh`
- Existing Myriad/qsub wrappers: `scripts/qsub_myriad_stage2_*.sh`

Existing outputs were found under `qamel/outputs/`. Usable files include:

- `qamel/outputs/studies/lq_seed12345_progressive/`
- `qamel/outputs/studies/cpu_swapaware_seed12345/`
- `qamel/outputs/studies/lq_seed12345_progressive/diagnostics/eval_ablations/`
- `qamel/outputs/head_to_head/smoke.json`

No complete n=6 publication run was found locally.

## Local Sanity Checks Run

These were local smoke checks only, not publishable learning results.

```bash
python -c "import qamel.dqn; import qamel.utils; import scripts.head_to_head; from qamel.utils import test_chain_progress_potential; test_chain_progress_potential(); print('ok')"
python -m py_compile qamel/dqn.py qamel/utils.py scripts/train_qamel.py scripts/evaluate_qamel.py scripts/head_to_head.py
python scripts/train_qamel.py --n 3 --pgen 0.4 --pswap 0.7 --obs_mode counter_exposed_plus_ready --seed 20260611 --reward_mode base --model_tag codex_publication_smoke --train_episodes 5 --checkpoint_every 5 --log_every 1 --force_train
python scripts/evaluate_qamel.py --n 3 --pgen 0.4 --pswap 0.7 --obs_mode counter_exposed_plus_ready --seed 20260611 --reward_mode base --model_tag codex_publication_smoke --eval_episodes 5 --max_actions 30 --eval_epsilon 0.0
python scripts/head_to_head.py --run-name dqn_n3_pgen0.4_pswap0.7_codex_publication_smoke --episodes 3 --seeds 20260611 20260612 --max-actions 30 --eval-epsilon 0.0 --out qamel/outputs/head_to_head/codex_publication_smoke.json
```

Smoke results:

- n=3 5-episode DQN smoke eval: success rate 0.0, timeout rate 1.0. This is expected for a nearly untrained smoke checkpoint.
- n=3 head-to-head smoke: DQN policies failed, heuristic mean success 0.833 over two tiny seeds. This was used only to verify the harness.

## Myriad / qsub Status

`qsub` is not available on this machine. Hostname during inspection:

```text
eduroam-int-dhcp-97-145-197.ucl.ac.uk
```

No Myriad jobs were submitted and no job IDs exist for this run.

I added ready-to-run wrappers:

- `scripts/myriad_publication_suite.sh`
- `scripts/qsub_publication_suite.sh`

Logs go to:

- `qamel/outputs/myriad_logs/`

The existing `scripts/run_stage2_study.sh` now accepts these optional environment flags:

- `DUELING=1`
- `DOUBLE_DQN=1`
- `PBRS=1`
- `PBRS_SCALE=<float>`
- `LR=<float>`
- `BATCH_SIZE=<int>`

Default behavior remains unchanged.

## Commands To Run On Myriad

Minimal n=5 five-seed canonical suite:

```bash
for S in 12345 23456 34567 45678 56789; do
  qsub -v EXPERIMENT_SET=n5_canonical,SEEDS=$S scripts/qsub_publication_suite.sh
done
```

Preliminary n=6 run:

```bash
qsub -v EXPERIMENT_SET=n6_prelim,SEEDS=12345 scripts/qsub_publication_suite.sh
```

Secondary Dueling + Double DQN comparison:

```bash
for S in 12345 23456 34567 45678 56789; do
  qsub -v EXPERIMENT_SET=n5_dueling_double,SEEDS=$S scripts/qsub_publication_suite.sh
done
```

Secondary Dueling + Double DQN + PBRS comparison:

```bash
for S in 12345 23456 34567 45678 56789; do
  qsub -v EXPERIMENT_SET=n5_dueling_double_pbrs,SEEDS=$S scripts/qsub_publication_suite.sh
done
```

Expected run tags:

- `pub_n5_canonical_dqn_n5_seed<S>`
- `pub_n6_prelim_dqn_n6_seed<S>`
- `pub_n5_dueling_double_dueling_double_n5_seed<S>`
- `pub_n5_dueling_double_pbrs_dueling_double_pbrs_n5_seed<S>`

Expected study directories:

- `qamel/outputs/studies/<run_tag>/`

## Figure and Table Generation

I added:

```bash
python scripts/make_publication_figures.py
```

It reads existing real outputs and writes:

- Figures: `figures/publication/`
- Tables: `qamel/outputs/analysis/`

The script keeps all rows in inventory tables but, by default, only includes rows with at least 50 evaluation episodes in figures.

Generated figures:

- `figures/publication/success_rate_vs_budget.pdf`
- `figures/publication/success_rate_vs_budget.png`
- `figures/publication/mean_return_vs_budget.pdf`
- `figures/publication/mean_return_vs_budget.png`
- `figures/publication/mean_steps_vs_budget.pdf`
- `figures/publication/mean_steps_vs_budget.png`
- `figures/publication/generation_attempts_vs_budget.pdf`
- `figures/publication/generation_attempts_vs_budget.png`
- `figures/publication/swap_attempts_vs_budget.pdf`
- `figures/publication/swap_attempts_vs_budget.png`
- `figures/publication/truncated_fraction_vs_budget.pdf`
- `figures/publication/truncated_fraction_vs_budget.png`
- `figures/publication/combined_budget_summary.pdf`
- `figures/publication/combined_budget_summary.png`
- `figures/publication/ablation_success_comparison.pdf`
- `figures/publication/ablation_success_comparison.png`
- `figures/publication/head_to_head_policy_comparison.pdf`
- `figures/publication/head_to_head_policy_comparison.png`

Generated tables:

- `qamel/outputs/analysis/budget_results.csv`
- `qamel/outputs/analysis/budget_results_aggregate.csv`
- `qamel/outputs/analysis/budget_results_figure_subset.csv`
- `qamel/outputs/analysis/evaluation_inventory.csv`
- `qamel/outputs/analysis/ablation_results.csv`
- `qamel/outputs/analysis/head_to_head_results.csv`
- `qamel/outputs/analysis/head_to_head_aggregate.csv`
- `qamel/outputs/analysis/head_to_head_figure_subset.csv`
- `qamel/outputs/analysis/publication_artifacts_manifest.csv`

## Main Existing Numerical Results

These are existing/local outputs, not a new full publication suite.

Budget diagnostic data used in figures:

| Study | Budget | Eval episodes | Success rate | Mean return | Mean steps | Truncated fraction |
|---|---:|---:|---:|---:|---:|---:|
| `lq_seed12345_progressive` | 50 | 500 | 0.354 | -94.524 | 69.288 | 0.646 |
| `lq_seed12345_progressive` | 200 | 500 | 0.008 | -196.725 | 99.270 | 0.992 |
| `lq_seed12345_progressive` | 500 | 500 | 0.210 | -121.110 | 81.344 | 0.790 |
| `lq_seed12345_progressive` | 1000 | 500 | 0.000 | -199.000 | 100.000 | 1.000 |
| `cpu_swapaware_seed12345` | 50 | 100 | 0.290 | -118.225 | 74.640 | 0.710 |
| `cpu_swapaware_seed12345` | 200 | 100 | 0.010 | -196.865 | 99.210 | 0.990 |
| `cpu_swapaware_seed12345` | 500 | 100 | 0.000 | -199.000 | 100.000 | 1.000 |

Head-to-head existing smoke data (`qamel/outputs/head_to_head/smoke.json`, three seeds, 200 episodes each):

| Policy | Mean success rate | Seed-level SE | Mean steps | Mean return | Truncated fraction |
|---|---:|---:|---:|---:|---:|
| `dqn_greedy` | 0.8217 | 0.0083 | 17.214 | 33.353 | 0.178 |
| `dqn_swapprefer` | 0.8917 | 0.0093 | 16.282 | 53.985 | 0.108 |
| `heuristic` | 0.8283 | 0.0073 | 36.763 | 19.055 | 0.172 |

Ablation highlights from existing checkpoint diagnostics:

- Checkpoint 700 baseline success: 0.186
- Checkpoint 700 no-op mask success: 0.404
- Checkpoint 700 refresh-block success: 0.238
- Checkpoint 700 swap-prefer success: 0.596
- Checkpoint 1000 baseline success: 0.000
- Checkpoint 1000 swap-prefer success: 0.680

## Caveats

- The existing budget outputs are diagnostic and unstable; they should not be used as final paper claims.
- The strongest-looking head-to-head data is labelled `smoke.json`, although it has three seeds and 200 episodes per seed. Treat it as preliminary until reproduced with the full matrix.
- No local n=6 publication result is available.
- No qsub jobs were submitted from this machine.
- The local n=3 smoke run created output files only to verify command paths.

## Next Recommended Experiment

Run the five-seed n=5 canonical Myriad matrix first. After it completes, run:

```bash
python scripts/make_publication_figures.py
```

Then inspect `qamel/outputs/analysis/budget_results_aggregate.csv` and decide whether to launch n=6 and Dueling/Double/PBRS variants.
