#!/bin/bash -l
#$ -N qamel_n7_pilot
#$ -l h_rt=48:00:00
#$ -l mem=24G
#$ -l gpu=1
# --- A100 PIN (edit if the job pends too long) ---------------------------------------
# `-ac allow=EF` selects Myriad's A100 node classes. At the n=7 / 70k-episode budget a
# V100 will very likely exceed the 48h wall, so pin A100. The canary below prints
# device_name so you can confirm an A100 was actually granted.
#$ -ac allow=EF
# -------------------------------------------------------------------------------------
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/n7_pilot_logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

# ==================================== WHAT THIS IS ====================================
# n=7 SINGLE-SEED PILOT (seed 202). Not an array. Must prove, before fanning out to 5 seeds:
#   (a) stable training through the 300-step final curriculum stage,
#   (b) greedy beats the re-baselined n=7 heuristic (checked AFTER, via head_to_head),
#   (c) projected walltime < 48h (via scripts/project_walltime.py).
#
# RESUMABLE BY DESIGN: this script does NOT pass --force_train, so a re-qsub of the SAME
# script resumes from checkpoints/latest.pt (model+target+optimizer+replay+global_step→epsilon
# +RNG+episode counter are all restored). First submission starts fresh (no checkpoint yet).
# This is the 48h-wall safety net: if the wall kills it mid-run, just qsub it again.
#
# Fixes from the n=6 post-mortem (root cause = COVERAGE: n=6 trained prefer_swap=False, so the
# behavior policy almost never sampled ready-node swaps -> their Q* was never learned -> greedy
# kept avoiding them):
#   * --swap_guidance ready_node     : TRAINING-ONLY soft guided exploration. Oversamples ready-node
#                                      swaps (same rule as eval) so their Q* gets learned, but keeps
#                                      non-swap actions sampled so their Q stays calibrated. Eval and
#                                      the TD target stay UNFILTERED -> no circularity, eval = greedy.
#   * --best_eval_max_actions 300    : best-eval is measured at the FINAL horizon (n=6 measured
#                                      at 100 while training to 150 — checkpoint selection bug).
#   * eps_decay 3.5M steps           : epsilon stays alive through every horizon expansion and
#                                      tapers into the final stage (n=6 hit floor at ~ep 6.7k).
#   * 4-stage curriculum 60/120/200/300 with gentler jumps than n=6.
# Reward is UNCHANGED (PBRS on, scale 1.0): PBRS is policy-invariant, so it is not the lever.
# =====================================================================================

# ============================ TUNABLES ============================
N="${N:-7}"
PGEN="${PGEN:-0.4}"
PSWAP="${PSWAP:-0.7}"
SEED="${SEED:-202}"
TRAIN_EPISODES="${TRAIN_EPISODES:-70000}"
MAX_ACTIONS="${MAX_ACTIONS:-300}"
EPS_DECAY="${EPS_DECAY:-3500000}"
SWAP_GUIDANCE_PROB="${SWAP_GUIDANCE_PROB:-0.5}"
CURRICULUM_BOUNDARIES="${CURRICULUM_BOUNDARIES:-15000,32000,48000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-60,120,200,300}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-2000}"
BEST_EVAL_EPISODES="${BEST_EVAL_EPISODES:-200}"
BEST_EVAL_MAX_ACTIONS="${BEST_EVAL_MAX_ACTIONS:-300}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RUN_TAG="${RUN_TAG:-stage0_fullstack_n7_pilot_s${SEED}}"
# =================================================================

REPO_ROOT="${REPO_ROOT:-/home/ucapgoz/Timing-Multiuser-Protocols}"
LOG_DIR="${REPO_ROOT}/qamel/outputs/n7_pilot_logs"
mkdir -p "${LOG_DIR}"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source "${VENV_PATH:-/home/ucapgoz/venvs/timing-multiuser/bin/activate}"

cd "${REPO_ROOT}"

RUN_NAME="dqn_n${N}_pgen${PGEN}_pswap${PSWAP}_${RUN_TAG}"
RUN_DIR="qamel/outputs/runs/${RUN_NAME}"

echo "host=$(hostname)"
echo "run_name=${RUN_NAME}"
echo "tunables: n=${N} train_episodes=${TRAIN_EPISODES} max_actions=${MAX_ACTIONS} lr=${LR} "\
"eps_decay=${EPS_DECAY} curriculum=${CURRICULUM_BOUNDARIES}/${CURRICULUM_STEPS} swap_guidance=ready_node/${SWAP_GUIDANCE_PROB}"
if [ -f "${RUN_DIR}/checkpoints/latest.pt" ]; then
  echo "RESUME: found ${RUN_DIR}/checkpoints/latest.pt — training will continue from it."
else
  echo "FRESH: no latest.pt — training starts from episode 0."
fi
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))  # CANARY: expect A100
PY

# NOTE: NO --force_train (so re-qsub resumes from latest.pt). NO checkpoint stripping / rm here —
# latest.pt is the resume anchor and must survive. best_eval.pt is stripped post-pilot in the
# RUN SHEET, not here. Curriculum is the generalized N-stage ladder (4 stages for n=7).
python scripts/train_qamel.py \
  --n "${N}" --pgen "${PGEN}" --pswap "${PSWAP}" \
  --obs_mode counter_exposed_plus_ready \
  --seed "${SEED}" --reward_mode base \
  --train_episodes "${TRAIN_EPISODES}" --max_actions "${MAX_ACTIONS}" \
  --eps_decay_steps "${EPS_DECAY}" \
  --use_curriculum --curriculum_boundaries "${CURRICULUM_BOUNDARIES}" --curriculum_steps "${CURRICULUM_STEPS}" \
  --swap_guidance ready_node --swap_guidance_prob "${SWAP_GUIDANCE_PROB}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" --log_every 200 \
  --best_eval_every "${BEST_EVAL_EVERY}" --best_eval_episodes "${BEST_EVAL_EPISODES}" \
  --best_eval_max_actions "${BEST_EVAL_MAX_ACTIONS}" \
  --double-dqn --dueling --pbrs --pbrs-scale 1.0 \
  --lr "${LR}" --batch_size "${BATCH_SIZE}" \
  --model_tag "${RUN_TAG}"

echo "TRAINING COMPLETE for ${RUN_NAME}. Run head_to_head + strip per the RUN SHEET."
