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

# ============================ TUNABLES (retune after first seed) ============================
# Rationale for the first-pass n=6 values vs the n=5 defaults:
#   * MAX_ACTIONS 100 -> 150 : longer chain needs more steps to reach the end-to-end link.
#   * EPS_DECAY  10k  -> 30k : more env steps to terminal at n=6, so explore longer.
#   * CURRICULUM            : n=5 defaults only exercised two horizons; scale boundaries to
#                             the 40k budget so the agent sees 40 -> 80 -> 150 horizons.
#   * TRAIN_EPISODES 10k -> 40k : bigger state/action space (89 actions, width 768) needs more.
# After the first seed finishes, read truncated_fraction + mean steps and retune horizon /
# episodes / eps-decay before trusting the full set.
N="${N:-6}"
PGEN="${PGEN:-0.4}"
PSWAP="${PSWAP:-0.7}"
SEEDS=(202 303 404 505 606 707 808 909)
TRAIN_EPISODES="${TRAIN_EPISODES:-40000}"
MAX_ACTIONS="${MAX_ACTIONS:-150}"
EPS_DECAY="${EPS_DECAY:-30000}"
CURRICULUM_BOUNDARIES="${CURRICULUM_BOUNDARIES:-15000,30000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-40,80,150}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-2000}"
BEST_EVAL_EPISODES="${BEST_EVAL_EPISODES:-200}"
LR="${LR:-5e-4}"
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
RUN_TAG="stage0_fullstack_n6_s${SEED}"
RUN_NAME="dqn_n${N}_pgen${PGEN}_pswap${PSWAP}_${RUN_TAG}"
RUN_DIR="qamel/outputs/runs/${RUN_NAME}"
BEST_EVAL="${RUN_DIR}/checkpoints/best_eval.pt"
LEGACY_MODEL="qamel/outputs/models/${RUN_NAME}.pt"

echo "host=$(hostname)"
echo "task_id=${TASK_ID}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
echo "tunables: n=${N} train_episodes=${TRAIN_EPISODES} max_actions=${MAX_ACTIONS} eps_decay=${EPS_DECAY} curriculum=${CURRICULUM_BOUNDARIES}/${CURRICULUM_STEPS}"
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
