#!/bin/bash -l
#$ -N qamel_n7_base
#$ -l h_rt=36:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -t 1-3
# --- A100 PIN (edit if jobs pend too long) -------------------------------------------
# `-ac allow=EF` selects Myriad's A100 node classes. At the n=7 budget a V100 will very
# likely exceed h_rt, so pin A100. If a task pends excessively, confirm the current
# node-type token via Myriad docs / rc-support and update this line. The first log lines
# print device_name, so the canary confirms whether an A100 was actually granted.
#$ -ac allow=EF
# -------------------------------------------------------------------------------------
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/n7_baseline_logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

# ==================================== WHAT THIS IS ====================================
# DELIBERATE BASELINE / CONTROL: the flat-MLP (dueling) Q-head over the full, UNfactored
# n=7 action set (~233 actions). It is EXPECTED to rank poorly / fail to clear the
# heuristic at this action-count — a weak, non-ranking result here is the intended control
# that motivates the factored action head. Do not treat under-performance as a bug.
#
# No correctness sentinel exists at n=7: the tabular enumeration blows up, so there is no
# cheap regression check here. Correctness rests on the n=5 --max_actions regression and on
# watching the training curves (best-eval SR rising, truncated_fraction bounded).
# Kept cheap: only 3 seeds.
# =====================================================================================

# ============================ TUNABLES (v2 — mirror the n=6 stability fixes) =================
# Apply the SAME v2 fixes as n=6 so this stays a clean control: a weak n=7 result must be
# attributable to the flat head over 233 actions, NOT to under-tuning (else a reviewer dismisses
# it). The ε-at-floor problem is worse here (60k budget): v1 eps_decay=45k floored ε by ~ep1150.
#   * EPS_DECAY 45k -> 450k : ramp over ~first 20% of the run (scaled to the larger n=7 budget).
#   * LR 5e-4 -> 2e-4       : same Q-overestimation fix as n=6.
#   * BEST_EVAL_MAX_ACTIONS 100 -> 200 : align best-eval horizon with the n=7 training/gate (200).
#   * RUN_TAG -> *_v2_*     : fresh tag, no collision with the v1 baseline runs.
N="${N:-7}"
PGEN="${PGEN:-0.4}"
PSWAP="${PSWAP:-0.7}"
SEEDS=(202 303 404)
TRAIN_EPISODES="${TRAIN_EPISODES:-60000}"
MAX_ACTIONS="${MAX_ACTIONS:-200}"
EPS_DECAY="${EPS_DECAY:-450000}"
CURRICULUM_BOUNDARIES="${CURRICULUM_BOUNDARIES:-20000,40000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-50,100,200}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-3000}"
BEST_EVAL_EPISODES="${BEST_EVAL_EPISODES:-200}"
BEST_EVAL_MAX_ACTIONS="${BEST_EVAL_MAX_ACTIONS:-200}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
EVAL_SEEDS="${EVAL_SEEDS:-12345 23456 34567 45678 56789}"
# ===========================================================================================

REPO_ROOT="${REPO_ROOT:-/home/ucapgoz/Timing-Multiuser-Protocols}"
LOG_DIR="${REPO_ROOT}/qamel/outputs/n7_baseline_logs"
mkdir -p "${LOG_DIR}"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source "${VENV_PATH:-/home/ucapgoz/venvs/timing-multiuser/bin/activate}"

cd "${REPO_ROOT}"

TASK_ID="${SGE_TASK_ID:-1}"
SEED="${SEEDS[$((TASK_ID - 1))]}"
RUN_TAG="flathead_n7_v2_s${SEED}"
RUN_NAME="dqn_n${N}_pgen${PGEN}_pswap${PSWAP}_${RUN_TAG}"
RUN_DIR="qamel/outputs/runs/${RUN_NAME}"
BEST_EVAL="${RUN_DIR}/checkpoints/best_eval.pt"
LEGACY_MODEL="qamel/outputs/models/${RUN_NAME}.pt"

echo "host=$(hostname)"
echo "task_id=${TASK_ID}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "tunables: n=${N} train_episodes=${TRAIN_EPISODES} max_actions=${MAX_ACTIONS} eps_decay=${EPS_DECAY} lr=${LR} best_eval_max_actions=${BEST_EVAL_MAX_ACTIONS} curriculum=${CURRICULUM_BOUNDARIES}/${CURRICULUM_STEPS}"
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))  # CANARY: expect A100
PY

# Skip a seed that already produced a stripped (weights-only) best_eval.pt.
if [ -f "${BEST_EVAL}" ]; then
  BEST_BYTES=$(python - <<PY
import os
p = "${BEST_EVAL}"
print(os.path.getsize(p) if os.path.exists(p) else 0)
PY
)
  if [ "${BEST_BYTES}" -lt $((20 * 1024 * 1024)) ]; then
    echo "seed ${SEED} already has stripped best_eval.pt (${BEST_BYTES} bytes); skipping"
    exit 0
  fi
fi

# Fresh tag + --force_train ONLY (never resume: width auto-scales to 1024 at n=7).
python scripts/train_qamel.py \
  --n "${N}" --pgen "${PGEN}" --pswap "${PSWAP}" \
  --obs_mode counter_exposed_plus_ready \
  --seed "${SEED}" --reward_mode base \
  --train_episodes "${TRAIN_EPISODES}" --max_actions "${MAX_ACTIONS}" \
  --eps_decay_steps "${EPS_DECAY}" \
  --use_curriculum --curriculum_boundaries "${CURRICULUM_BOUNDARIES}" --curriculum_steps "${CURRICULUM_STEPS}" \
  --checkpoint_every 0 --log_every 200 \
  --best_eval_every "${BEST_EVAL_EVERY}" --best_eval_episodes "${BEST_EVAL_EPISODES}" \
  --best_eval_max_actions "${BEST_EVAL_MAX_ACTIONS}" \
  --double-dqn --dueling --pbrs --pbrs-scale 1.0 \
  --lr "${LR}" --batch_size "${BATCH_SIZE}" \
  --force_train \
  --model_tag "${RUN_TAG}"

# Drop replay-embedding artifacts (keep only best_eval.pt, which we strip next).
rm -f "${RUN_DIR}/model.pt" "${LEGACY_MODEL}" "${RUN_DIR}/checkpoints/latest.pt" "${RUN_DIR}"/checkpoints/episode_*.pt

python - <<PY
import os
import torch

p = "${BEST_EVAL}"
ck = torch.load(p, map_location="cpu", weights_only=False)
for key in (
    "replay_buffer_state",
    "replay_buffer",
    "optimizer_state",
    "optimizer",
    "target_model_state",
    "target_net_state",
    "target_net",
):
    ck.pop(key, None)
torch.save(ck, p)
size = os.path.getsize(p)
print(f"stripped_best_eval_bytes={size}")
if size >= 20 * 1024 * 1024:
    raise SystemExit(f"stripped best_eval.pt too large: {size} bytes")
PY

# Paired evaluation at the SAME horizon as training (--max-actions ${MAX_ACTIONS}).
python scripts/head_to_head.py \
  --run-name "${RUN_NAME}" \
  --episodes "${EVAL_EPISODES}" --seeds ${EVAL_SEEDS} \
  --max-actions "${MAX_ACTIONS}" --eval-epsilon 0.0 \
  --out "${RUN_DIR}/head_to_head.json"

python - <<PY
import json
import os

best = "${BEST_EVAL}"
h2h = "${RUN_DIR}/head_to_head.json"
print("final_best_eval_bytes", os.path.getsize(best))
print("head_to_head_net_arch", json.load(open(h2h))["config"]["net_arch"])
PY
