#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_TAG="lq_n6_seed12345_intermediate"
export SEED="12345"
export N="6"
export PGEN="0.4"
export PSWAP="0.7"
export OBS_MODE="counter_exposed_plus_ready"
export REWARD_MODE="base"
export EVAL_EPISODES="500"
export MAX_ACTIONS="100"
export EVAL_EPSILON="0.0"
export BUDGETS="100 200 500"
export CHECKPOINT_EVERY="100"
export LOG_EVERY="50"
export USE_CURRICULUM="1"
export STUDY_ROOT="qamel/outputs/studies"
export STUDY_FORCE_TRAIN="${STUDY_FORCE_TRAIN:-0}"
export ALLOW_ARCHIVE_OVERWRITE="${ALLOW_ARCHIVE_OVERWRITE:-0}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/run_stage2_study.sh"
