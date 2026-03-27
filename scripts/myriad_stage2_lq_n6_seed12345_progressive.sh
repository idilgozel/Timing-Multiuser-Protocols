#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_TAG="lq_n6_seed12345_progressive"
export SEED="12345"
export N="6"
export PGEN="0.4"
export PSWAP="0.7"
export OBS_MODE="counter_exposed_plus_ready"
export REWARD_MODE="base"
export EVAL_EPISODES="500"
export MAX_ACTIONS="100"
export EVAL_EPSILON="0.0"
export BUDGETS="10000 50000 100000 200000"
export CHECKPOINT_EVERY="1000"
export LOG_EVERY="100"
export STUDY_ROOT="qamel/outputs/studies"
export USE_CURRICULUM="1"
export STUDY_FORCE_TRAIN="${STUDY_FORCE_TRAIN:-0}"
export ALLOW_ARCHIVE_OVERWRITE="${ALLOW_ARCHIVE_OVERWRITE:-0}"

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/run_stage2_study.sh"
