#!/bin/bash -l
#$ -N qamel_n6_esc
#$ -l h_rt=24:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -t 1-8
# --- A100 PIN (edit if jobs pend too long) -------------------------------------------
# `-ac allow=EF` selects Myriad's A100 node classes. The n=5 array landed one task on a
# slower V100 (wall-time / truncation risk at the longer n=6 budget), so pin A100 here.
# If a task pends excessively, confirm the current node-type token via Myriad docs or
# rc-support and update this line (or comment it out to accept any GPU). The first log
# lines print device_name, so the canary confirms whether an A100 was actually granted.
#$ -ac allow=EF
# -------------------------------------------------------------------------------------
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/n6_escalation_logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

# ============================ TUNABLES (v2 retune after first canary) =======================
# v1 (eps_decay=30k, lr=5e-4, best_eval horizon=100) showed learn-then-collapse at n=6:
# s202 peaked 0.62 greedy @ ep6k then fell to 0.25; s303 stored an early 0.68 peak @ ep4k,
# then fell to ~0.01--0.02 by ep10k--16k. Diagnosis: ε pinned at the 0.05 floor for ~98% of
# the run (30k env steps ~= ep900 of 40k) + too-high LR at width 768 / 89 actions ->
# Q-overestimation oscillation with no exploration left to recover.
# v2 changes (change ONLY the diagnosed levers; keep horizon/curriculum/episodes fixed so the
# fix is attributable):
#   * EPS_DECAY 30k -> 250k : ramp ε down over the first ~20% of the run (~ep8k), not ~ep900,
#                             so a dipping policy can re-explore back out of the collapse.
#   * LR 5e-4 -> 2e-4       : canonical fix for learn-then-collapse / moving-target oscillation.
#   * BEST_EVAL_MAX_ACTIONS 100 -> 150 : BUG FIX. best-eval ran at horizon 100 while training and
#                             the gate use 150, so "best" was selected at the wrong horizon
#                             (failing seeds read mean_steps ~98 = truncation at 100, not 150).
#   * RUN_TAG -> *_v2_*     : fresh tag so this relaunch never collides with / resumes the v1
#                             runs and their frozen best_eval.pt artifacts stay intact.
# Next lever if v2 still collapses (do NOT stack now): --prefer_swap_when_ready_train, or
# soften the target sync 1000 -> 2000 (needs a CLI flag; not exposed yet), or the factored head.
N="${N:-6}"
PGEN="${PGEN:-0.4}"
PSWAP="${PSWAP:-0.7}"
SEEDS=(202 303 404 505 606 707 808 909)
TRAIN_EPISODES="${TRAIN_EPISODES:-40000}"
MAX_ACTIONS="${MAX_ACTIONS:-150}"
EPS_DECAY="${EPS_DECAY:-250000}"
CURRICULUM_BOUNDARIES="${CURRICULUM_BOUNDARIES:-15000,30000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-40,80,150}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-2000}"
BEST_EVAL_EPISODES="${BEST_EVAL_EPISODES:-200}"
BEST_EVAL_MAX_ACTIONS="${BEST_EVAL_MAX_ACTIONS:-150}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-1000}"
EVAL_SEEDS="${EVAL_SEEDS:-12345 23456 34567 45678 56789}"
# ===========================================================================================

REPO_ROOT="${REPO_ROOT:-/home/ucapgoz/Timing-Multiuser-Protocols}"
LOG_DIR="${REPO_ROOT}/qamel/outputs/n6_escalation_logs"
mkdir -p "${LOG_DIR}"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source "${VENV_PATH:-/home/ucapgoz/venvs/timing-multiuser/bin/activate}"

cd "${REPO_ROOT}"

TASK_ID="${SGE_TASK_ID:-1}"
SEED="${SEEDS[$((TASK_ID - 1))]}"
RUN_TAG="stage0_fullstack_n6_v2_s${SEED}"
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

# Fresh tag + --force_train ONLY (never resume: width auto-scales to 768 at n=6, and the
# width-mismatch trap bites on resume). --checkpoint_every 0 keeps replay-bloated snapshots
# off disk; we strip best_eval.pt to weights-only below.
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
