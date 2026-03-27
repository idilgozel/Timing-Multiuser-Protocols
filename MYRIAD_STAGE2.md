# Stage 2 Study Workflow

This note covers the staged Stage 2 budget study for the current executable DQN scope.

Scientific benchmark remains fixed:

- `n=5`
- `pgen=0.4`
- `pswap=0.7`
- `obs_mode=counter_exposed_plus_ready`
- `reward_mode=base`
- `eval_episodes=500`
- `max_actions=100`
- `eval_epsilon=0.0`

## Local pilot

Workflow-validation only. Not a learning-quality claim.

Wrapper:

```bash
bash scripts/local_pilot_stage2_lq_seed12345_progressive.sh
```

Equivalent explicit command:

```bash
RUN_TAG=lq_seed12345_progressive \
SEED=12345 \
BUDGETS="50 200 500 1000" \
CHECKPOINT_EVERY=50 \
LOG_EVERY=50 \
STUDY_FORCE_TRAIN=1 \
bash scripts/run_stage2_study.sh
```

Measured local behavior on this CPU-only environment:

- `50` episodes took about `14.5` minutes
- the local pilot is suitable for workflow validation
- it is not a fast path for substantive learning-quality measurement

This writes:

- run-local training artifacts:
  - `qamel/outputs/runs/dqn_n5_pgen0.4_pswap0.7_lq_seed12345_progressive/`
- archived pilot study outputs:
  - `qamel/outputs/studies/lq_seed12345_progressive/budget_<BUDGET>/`
- cross-budget summary:
  - `qamel/outputs/studies/lq_seed12345_progressive/cross_budget_summary.csv`

## Myriad run

The Myriad-friendly wrapper is:

```bash
bash scripts/myriad_stage2_lq_seed12345_progressive.sh
```

Edit the module and environment activation lines in that script for the actual cluster setup before launch.

The wrapper uses budgets:

- `10000`
- `50000`
- `100000`
- `200000`

and archives each budget under:

- `qamel/outputs/studies/lq_seed12345_progressive/budget_<BUDGET>/`

## Output structure

Each archived budget directory contains:

- `model.pt`
- `latest.pt`
- `metrics.csv`
- `train.log`
- `config.json`
- `eval.csv`
- `eval_summary.json`

The cross-budget summary CSV contains:

- `budget`
- `success_rate`
- `mean_return`
- `mean_steps`
- `mean_ent_attempt_max`
- `mean_swap_attempt_max`
- `truncated_fraction`

## Overwrite behavior

The study script refuses to overwrite a non-empty archived budget directory by default.

To allow overwriting archived budget directories:

```bash
ALLOW_ARCHIVE_OVERWRITE=1 bash scripts/run_stage2_study.sh
```

To intentionally restart training from scratch for a new run directory:

```bash
STUDY_FORCE_TRAIN=1 bash scripts/run_stage2_study.sh
```

`STUDY_FORCE_TRAIN=1` only passes `--force_train` when the run directory does not already exist. Resume behavior is preserved for later budgets in the same staged run.
