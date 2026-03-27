#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N="${N:-5}"
PGEN="${PGEN:-0.4}"
PSWAP="${PSWAP:-0.7}"
OBS_MODE="${OBS_MODE:-counter_exposed_plus_ready}"
REWARD_MODE="${REWARD_MODE:-base}"
EVAL_EPISODES="${EVAL_EPISODES:-500}"
MAX_ACTIONS="${MAX_ACTIONS:-100}"
EVAL_EPSILON="${EVAL_EPSILON:-0.0}"
SEED="${SEED:-12345}"
RUN_TAG="${RUN_TAG:-lq_seed12345_progressive}"
BUDGETS="${BUDGETS:-10000 50000 100000 200000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"
BEST_EVAL_EVERY="${BEST_EVAL_EVERY:-0}"
BEST_EVAL_EPISODES="${BEST_EVAL_EPISODES:-500}"
BEST_EVAL_MAX_ACTIONS="${BEST_EVAL_MAX_ACTIONS:-100}"
BEST_EVAL_SEED="${BEST_EVAL_SEED:-${SEED}}"
COUNTER_NORM="${COUNTER_NORM:-20.0}"
SWAP_READY_BONUS="${SWAP_READY_BONUS:-0.5}"
PREFER_SWAP_WHEN_READY_TRAIN="${PREFER_SWAP_WHEN_READY_TRAIN:-0}"
USE_CURRICULUM="${USE_CURRICULUM:-0}"
TRAIN_DEBUG_FLAG="${TRAIN_DEBUG_FLAG:-}"
EVAL_DEBUG_FLAG="${EVAL_DEBUG_FLAG:-}"
EVAL_DIAG_FLAG="${EVAL_DIAG_FLAG:-}"
STUDY_ROOT="${STUDY_ROOT:-qamel/outputs/studies}"
ALLOW_ARCHIVE_OVERWRITE="${ALLOW_ARCHIVE_OVERWRITE:-0}"
STUDY_FORCE_TRAIN="${STUDY_FORCE_TRAIN:-0}"

RUN_NAME="dqn_n${N}_pgen${PGEN}_pswap${PSWAP}_${RUN_TAG}"
RUN_DIR="qamel/outputs/runs/${RUN_NAME}"
STUDY_DIR="${STUDY_ROOT}/${RUN_TAG}"
SUMMARY_CSV="${STUDY_DIR}/cross_budget_summary.csv"

mkdir -p "${STUDY_DIR}"

rebuild_summary_csv() {
  python - <<'PY' "${STUDY_DIR}" "${SUMMARY_CSV}"
import csv
import json
import os
import sys
from pathlib import Path

study_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for budget_dir in sorted(study_dir.glob("budget_*"), key=lambda p: int(p.name.split("_", 1)[1])):
    summary_path = budget_dir / "eval_summary.json"
    if not summary_path.exists():
        continue
    payload = json.loads(summary_path.read_text())
    rows.append(
        {
            "budget": int(budget_dir.name.split("_", 1)[1]),
            "success_rate": payload["success_rate"],
            "mean_return": payload["total_return_mean"],
            "mean_steps": payload["steps_mean"],
            "mean_ent_attempt_max": payload["ent_mean"],
            "mean_swap_attempt_max": payload["swap_mean"],
            "truncated_fraction": payload.get("timeout_rate"),
        }
    )

with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "budget",
            "success_rate",
            "mean_return",
            "mean_steps",
            "mean_ent_attempt_max",
            "mean_swap_attempt_max",
            "truncated_fraction",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY
}

echo "Stage 2 study starting"
echo "run_tag=${RUN_TAG}"
echo "run_dir=${RUN_DIR}"
echo "study_dir=${STUDY_DIR}"
echo "budgets=${BUDGETS}"

for BUDGET in ${BUDGETS}; do
  BUDGET_DIR="${STUDY_DIR}/budget_${BUDGET}"
  EVAL_NAME="eval_episodes${EVAL_EPISODES}_max${MAX_ACTIONS}_eps${EVAL_EPSILON}_seed${SEED}"
  EVAL_DIR="${RUN_DIR}/evaluations"
  EVAL_CSV="${EVAL_DIR}/${EVAL_NAME}.csv"
  EVAL_SUMMARY="${EVAL_DIR}/${EVAL_NAME}_summary.json"

  if [[ -d "${BUDGET_DIR}" ]] && [[ -n "$(find "${BUDGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "${ALLOW_ARCHIVE_OVERWRITE}" != "1" ]]; then
    echo "Refusing to overwrite existing non-empty archive directory: ${BUDGET_DIR}" >&2
    echo "Set ALLOW_ARCHIVE_OVERWRITE=1 to overwrite archived budget outputs." >&2
    exit 1
  fi

  mkdir -p "${BUDGET_DIR}"

  TRAIN_ARGS=(
    python scripts/train_qamel.py
    --n "${N}"
    --pgen "${PGEN}"
    --pswap "${PSWAP}"
    --obs_mode "${OBS_MODE}"
    --reward_mode "${REWARD_MODE}"
    --seed "${SEED}"
    --model_tag "${RUN_TAG}"
    --train_episodes "${BUDGET}"
    --checkpoint_every "${CHECKPOINT_EVERY}"
    --log_every "${LOG_EVERY}"
    --best_eval_every "${BEST_EVAL_EVERY}"
    --best_eval_episodes "${BEST_EVAL_EPISODES}"
    --best_eval_max_actions "${BEST_EVAL_MAX_ACTIONS}"
    --best_eval_seed "${BEST_EVAL_SEED}"
    --swap_ready_bonus "${SWAP_READY_BONUS}"
  )
  if [[ "${PREFER_SWAP_WHEN_READY_TRAIN}" == "1" ]]; then
    TRAIN_ARGS+=(--prefer_swap_when_ready_train)
  fi
  if [[ "${USE_CURRICULUM}" == "1" ]]; then
    TRAIN_ARGS+=(--use_curriculum)
  fi
  if [[ -n "${TRAIN_DEBUG_FLAG}" ]]; then
    TRAIN_ARGS+=("${TRAIN_DEBUG_FLAG}")
  fi
  if [[ "${STUDY_FORCE_TRAIN}" == "1" ]] && [[ ! -d "${RUN_DIR}" ]]; then
    TRAIN_ARGS+=(--force_train)
  fi

  echo "Training to budget=${BUDGET}"
  "${TRAIN_ARGS[@]}"

  echo "Evaluating budget=${BUDGET}"
  EVAL_ARGS=(
    python scripts/evaluate_qamel.py
    --n "${N}"
    --pgen "${PGEN}"
    --pswap "${PSWAP}"
    --obs_mode "${OBS_MODE}"
    --reward_mode "${REWARD_MODE}"
    --seed "${SEED}"
    --model_tag "${RUN_TAG}"
    --eval_episodes "${EVAL_EPISODES}"
    --max_actions "${MAX_ACTIONS}"
    --eval_epsilon "${EVAL_EPSILON}"
    --counter_norm "${COUNTER_NORM}"
    --swap_ready_bonus "${SWAP_READY_BONUS}"
  )
  if [[ -n "${EVAL_DEBUG_FLAG}" ]]; then
    EVAL_ARGS+=("${EVAL_DEBUG_FLAG}")
  fi
  if [[ -n "${EVAL_DIAG_FLAG}" ]]; then
    EVAL_ARGS+=("${EVAL_DIAG_FLAG}")
  fi
  "${EVAL_ARGS[@]}"

  if [[ ! -f "${RUN_DIR}/model.pt" ]]; then
    echo "Missing run-local model: ${RUN_DIR}/model.pt" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/checkpoints/latest.pt" ]]; then
    echo "Missing latest checkpoint: ${RUN_DIR}/checkpoints/latest.pt" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/metrics.csv" ]]; then
    echo "Missing metrics.csv: ${RUN_DIR}/metrics.csv" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/train.log" ]]; then
    echo "Missing train.log: ${RUN_DIR}/train.log" >&2
    exit 1
  fi
  if [[ ! -f "${RUN_DIR}/config.json" ]]; then
    echo "Missing config.json: ${RUN_DIR}/config.json" >&2
    exit 1
  fi
  if [[ ! -f "${EVAL_CSV}" ]]; then
    echo "Missing evaluation CSV: ${EVAL_CSV}" >&2
    exit 1
  fi
  if [[ ! -f "${EVAL_SUMMARY}" ]]; then
    echo "Missing evaluation summary: ${EVAL_SUMMARY}" >&2
    exit 1
  fi

  cp "${RUN_DIR}/model.pt" "${BUDGET_DIR}/model.pt"
  cp "${RUN_DIR}/checkpoints/latest.pt" "${BUDGET_DIR}/latest.pt"
  cp "${RUN_DIR}/metrics.csv" "${BUDGET_DIR}/metrics.csv"
  cp "${RUN_DIR}/train.log" "${BUDGET_DIR}/train.log"
  cp "${RUN_DIR}/config.json" "${BUDGET_DIR}/config.json"
  cp "${EVAL_CSV}" "${BUDGET_DIR}/eval.csv"
  cp "${EVAL_SUMMARY}" "${BUDGET_DIR}/eval_summary.json"
  if [[ -f "${RUN_DIR}/checkpoints/best_eval.pt" ]]; then
    cp "${RUN_DIR}/checkpoints/best_eval.pt" "${BUDGET_DIR}/best_eval.pt"
  fi
  if [[ -f "${RUN_DIR}/checkpoints/best_eval_metrics.json" ]]; then
    cp "${RUN_DIR}/checkpoints/best_eval_metrics.json" "${BUDGET_DIR}/best_eval_metrics.json"
  fi

  rebuild_summary_csv
  echo "Archived budget=${BUDGET} to ${BUDGET_DIR}"
done

echo "Cross-budget summary written to ${SUMMARY_CSV}"
