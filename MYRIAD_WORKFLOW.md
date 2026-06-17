# Myriad Workflow Notes

Lab notebook date: 2026-06-17

This note records the working UCL Myriad setup for the Qamel / Timing-Multiuser-Protocols runs. It is meant to be practical: use it before submitting Stage-0, Stage-2, publication-suite, or head-to-head jobs.

## Login And Repository

```bash
ssh ucapgoz@myriad.rc.ucl.ac.uk
cd /home/ucapgoz/Timing-Multiuser-Protocols
```

The repository path used by the qsub wrappers is:

```bash
/home/ucapgoz/Timing-Multiuser-Protocols
```

## Environment Setup

Two virtual environments exist:

```text
/home/ucapgoz/venvs/timing-multiuser
/home/ucapgoz/venvs/dqn-env
```

Use `timing-multiuser`. Do not use `dqn-env` for current jobs; it failed with:

```text
python: error while loading shared libraries: libpython3.11.so.1.0
```

Important: `timing-multiuser` only sees PyTorch after the Myriad PyTorch module is loaded. If you activate the venv first in a plain login shell, `import torch` fails.

Use this sequence:

```bash
deactivate 2>/dev/null || true

module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu

source /home/ucapgoz/venvs/timing-multiuser/bin/activate
```

Validate the environment:

```bash
which python
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import qamel.dqn; import qamel.utils; import scripts.head_to_head; print('imports ok')"
```

On the login node, `torch.cuda.is_available()` can print `False`. That is expected. Inside a GPU qsub job, the log should show `cuda_available True`.

The known-good login-node import check on 2026-06-17 was:

```text
Python 3.9.6
torch 2.1.0+cu121
imports ok
```

## Git Hygiene Before Submitting Jobs

Before qsub, make sure Myriad is running the current code:

```bash
git status --short
git pull --ff-only
python -c "import qamel.dqn; import qamel.utils; import scripts.head_to_head; print('imports ok after pull')"
```

If `git status --short` is not empty, stash before pulling. The Git version on Myriad did not accept `git stash push -u`, so use the older syntax:

```bash
git stash save -u "myriad pre-pull $(date +%Y%m%d-%H%M%S)"
git pull --ff-only
```

The previous Stage-0 escalation array failed because the Myriad checkout was stale. The old `scripts/train_qamel.py` rejected the new flags:

```text
--max_actions 100 --double-dqn --dueling --pbrs --pbrs-scale 1.0 --lr 5e-4 --batch_size 256
```

Treat any cluster result as suspect unless the log proves the job ran after a clean pull/import check.

## Submitting Jobs

### Stage-0 Escalation Array

After the environment and Git checks pass:

```bash
qsub scripts/myriad_stage0_escalation_array.sh
qstat -u "$USER"
```

This array uses task IDs 1-8 and writes logs to:

```text
qamel/outputs/stage0_escalation_logs/
```

Expected early log lines inside a valid GPU job:

```text
cuda_available True
device_count 1
device_name NVIDIA A100 80GB PCIe
```

### Publication Suite

The publication wrapper is:

```bash
qsub scripts/qsub_publication_suite.sh
```

It delegates to:

```text
scripts/myriad_publication_suite.sh
scripts/run_stage2_study.sh
```

Useful environment selectors include:

```bash
EXPERIMENT_SET=n5_canonical qsub scripts/qsub_publication_suite.sh
EXPERIMENT_SET=n6_prelim qsub scripts/qsub_publication_suite.sh
EXPERIMENT_SET=n5_dueling_double qsub scripts/qsub_publication_suite.sh
EXPERIMENT_SET=n5_dueling_double_pbrs qsub scripts/qsub_publication_suite.sh
```

The publication logs go to:

```text
qamel/outputs/myriad_logs/
```

### Older Stage-2 Wrappers

Older qsub wrappers still exist and write to:

```text
qamel/outputs/cluster_logs/
```

Examples:

```bash
qsub scripts/qsub_myriad_stage2_lq_n5_seed12345_progressive.sh
qsub scripts/qsub_myriad_stage2_lq_n6_seed12345_progressive.sh
qsub scripts/qsub_myriad_stage2_swapaware_seed12345.sh
```

Use these only when intentionally reproducing the older progressive-study workflow.

## Monitoring Jobs And Logs

Queue status:

```bash
qstat -u "$USER"
```

Find log directories:

```bash
find qamel/outputs -maxdepth 3 -type d \( -name "*log*" -o -name "*logs*" \) -print
```

Tail recent cluster logs:

```bash
tail -n 120 qamel/outputs/stage0_escalation_logs/*.out 2>/dev/null
tail -n 120 qamel/outputs/myriad_logs/*.out 2>/dev/null
tail -n 120 qamel/outputs/cluster_logs/*.out 2>/dev/null
```

Search for the important status lines:

```bash
grep -RInE "cuda_available|Success rate|Cross-budget|verdict|Traceback|error|size mismatch|unrecognized" \
  qamel/outputs/stage0_escalation_logs qamel/outputs/myriad_logs qamel/outputs/cluster_logs 2>/dev/null
```

For Stage-2 budget archives:

```bash
for f in qamel/outputs/studies/*/cross_budget_summary.csv; do
  echo
  echo "== $f =="
  cat "$f"
done
```

## Confirmed Results And Failures

### Confirmed n=5 Progressive Result

The strongest confirmed Myriad result is:

```text
run_tag: lq_n5_seed12345_progressive
n: 5
pgen: 0.4
pswap: 0.7
budget: 10000
success_rate: 0.904
mean_return: 58.618
mean_steps: 23.182
truncated_fraction: 0.096
eval episodes: 500
```

The same run has a checkpoint at `episode_011000.pt`, but no archived `budget_20000` evaluation. Cite the run as `10k complete`, not `20k complete`.

### Stage-0 Array Failure

The first Stage-0 escalation array did not test the science. It failed during argument parsing because the Myriad checkout was stale and did not include the newer CLI flags.

Fix before resubmission:

```bash
git status --short
git pull --ff-only
python -c "import qamel.dqn; import qamel.utils; import scripts.head_to_head; print('imports ok after pull')"
qsub scripts/myriad_stage0_escalation_array.sh
```

### n=6 Checkpoint Mismatch

Some old n=6 checkpoints use hidden width 512. The current n=6 model with `counter_exposed_plus_ready` uses hidden width 768. Resuming old n=6 runs can fail with:

```text
size mismatch for net.1.weight: checkpoint [512, 144], current [768, 144]
```

Use a fresh run tag for new n=6 jobs, or move the old run directory aside.

### swapaware Resume Failure

The old `swapaware_seed12345` run archived a 200-episode eval but later failed restoring RNG state:

```text
TypeError: RNG state must be a torch.ByteTensor
```

The current local training code converts restored RNG tensors before setting RNG state, but confirm this on Myriad after a clean pull before trusting resumed swapaware jobs.

## Quick Pre-Submission Checklist

```bash
cd /home/ucapgoz/Timing-Multiuser-Protocols

module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source /home/ucapgoz/venvs/timing-multiuser/bin/activate

git status --short
git pull --ff-only

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import qamel.dqn; import qamel.utils; import scripts.head_to_head; print('imports ok')"

qstat -u "$USER"
```

Then submit the desired qsub script.
