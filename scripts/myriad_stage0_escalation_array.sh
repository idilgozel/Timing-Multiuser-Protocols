#!/bin/bash -l
#$ -N qamel_s0_esc
#$ -l h_rt=06:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -t 1-8
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/stage0_escalation_logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ucapgoz/Timing-Multiuser-Protocols}"
LOG_DIR="${REPO_ROOT}/qamel/outputs/stage0_escalation_logs"
mkdir -p "${LOG_DIR}"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source "${VENV_PATH:-/home/ucapgoz/venvs/timing-multiuser/bin/activate}"

cd "${REPO_ROOT}"

SEEDS=(202 303 404 505 606 707 808 909)
TASK_ID="${SGE_TASK_ID:-1}"
SEED="${SEEDS[$((TASK_ID - 1))]}"
RUN_TAG="stage0_fullstack_s${SEED}"
RUN_NAME="dqn_n5_pgen0.4_pswap0.7_${RUN_TAG}"
RUN_DIR="qamel/outputs/runs/${RUN_NAME}"
BEST_EVAL="${RUN_DIR}/checkpoints/best_eval.pt"
LEGACY_MODEL="qamel/outputs/models/${RUN_NAME}.pt"

echo "host=$(hostname)"
echo "task_id=${TASK_ID}"
echo "seed=${SEED}"
echo "run_name=${RUN_NAME}"
python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

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

python scripts/train_qamel.py \
  --n 5 --pgen 0.4 --pswap 0.7 \
  --obs_mode counter_exposed_plus_ready \
  --seed "${SEED}" --reward_mode base \
  --train_episodes 10000 --max_actions 100 \
  --checkpoint_every 0 --log_every 100 \
  --best_eval_every 1000 --best_eval_episodes 200 \
  --double-dqn --dueling --pbrs --pbrs-scale 1.0 \
  --lr 5e-4 --batch_size 256 \
  --force_train \
  --model_tag "${RUN_TAG}"

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

python scripts/head_to_head.py \
  --run-name "${RUN_NAME}" \
  --episodes 1000 --seeds 12345 23456 34567 45678 56789 \
  --max-actions 100 --eval-epsilon 0.0 \
  --out "${RUN_DIR}/head_to_head.json"

python - <<PY
import json
import os

best = "${BEST_EVAL}"
h2h = "${RUN_DIR}/head_to_head.json"
print("final_best_eval_bytes", os.path.getsize(best))
print("head_to_head_net_arch", json.load(open(h2h))["config"]["net_arch"])
PY
