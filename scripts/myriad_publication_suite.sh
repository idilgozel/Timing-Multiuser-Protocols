#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT_SET="${EXPERIMENT_SET:-n5_canonical}"
SEEDS="${SEEDS:-12345}"
STUDY_ROOT="${STUDY_ROOT:-qamel/outputs/studies}"
ALLOW_ARCHIVE_OVERWRITE="${ALLOW_ARCHIVE_OVERWRITE:-0}"
STUDY_FORCE_TRAIN="${STUDY_FORCE_TRAIN:-0}"

COMMON_PGEN="${PGEN:-0.4}"
COMMON_PSWAP="${PSWAP:-0.7}"
COMMON_OBS_MODE="${OBS_MODE:-counter_exposed_plus_ready}"
COMMON_REWARD_MODE="${REWARD_MODE:-base}"
COMMON_MAX_ACTIONS="${MAX_ACTIONS:-100}"
COMMON_EVAL_EPSILON="${EVAL_EPSILON:-0.0}"

case "${EXPERIMENT_SET}" in
  n5_canonical)
    N="${N:-5}"
    BUDGETS="${BUDGETS:-10000 50000 100000 200000}"
    EVAL_EPISODES="${EVAL_EPISODES:-1000}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
    LOG_EVERY="${LOG_EVERY:-100}"
    USE_CURRICULUM="${USE_CURRICULUM:-1}"
    DUELING="${DUELING:-0}"
    DOUBLE_DQN="${DOUBLE_DQN:-0}"
    PBRS="${PBRS:-0}"
    VARIANT_TAG="dqn"
    ;;
  n6_prelim)
    N="${N:-6}"
    BUDGETS="${BUDGETS:-5000 10000 20000}"
    EVAL_EPISODES="${EVAL_EPISODES:-500}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
    LOG_EVERY="${LOG_EVERY:-100}"
    USE_CURRICULUM="${USE_CURRICULUM:-1}"
    DUELING="${DUELING:-0}"
    DOUBLE_DQN="${DOUBLE_DQN:-0}"
    PBRS="${PBRS:-0}"
    VARIANT_TAG="dqn"
    ;;
  n5_dueling_double)
    N="${N:-5}"
    BUDGETS="${BUDGETS:-10000 50000 100000 200000}"
    EVAL_EPISODES="${EVAL_EPISODES:-1000}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
    LOG_EVERY="${LOG_EVERY:-100}"
    USE_CURRICULUM="${USE_CURRICULUM:-1}"
    DUELING="${DUELING:-1}"
    DOUBLE_DQN="${DOUBLE_DQN:-1}"
    PBRS="${PBRS:-0}"
    VARIANT_TAG="dueling_double"
    ;;
  n5_dueling_double_pbrs)
    N="${N:-5}"
    BUDGETS="${BUDGETS:-10000 50000 100000 200000}"
    EVAL_EPISODES="${EVAL_EPISODES:-1000}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
    LOG_EVERY="${LOG_EVERY:-100}"
    USE_CURRICULUM="${USE_CURRICULUM:-1}"
    DUELING="${DUELING:-1}"
    DOUBLE_DQN="${DOUBLE_DQN:-1}"
    PBRS="${PBRS:-1}"
    PBRS_SCALE="${PBRS_SCALE:-1.0}"
    VARIANT_TAG="dueling_double_pbrs"
    ;;
  *)
    echo "Unknown EXPERIMENT_SET='${EXPERIMENT_SET}'" >&2
    echo "Expected one of: n5_canonical, n6_prelim, n5_dueling_double, n5_dueling_double_pbrs" >&2
    exit 2
    ;;
esac

for SEED in ${SEEDS}; do
  export N
  export PGEN="${COMMON_PGEN}"
  export PSWAP="${COMMON_PSWAP}"
  export OBS_MODE="${COMMON_OBS_MODE}"
  export REWARD_MODE="${COMMON_REWARD_MODE}"
  export MAX_ACTIONS="${COMMON_MAX_ACTIONS}"
  export EVAL_EPSILON="${COMMON_EVAL_EPSILON}"
  export EVAL_EPISODES
  export BUDGETS
  export CHECKPOINT_EVERY
  export LOG_EVERY
  export USE_CURRICULUM
  export DUELING
  export DOUBLE_DQN
  export PBRS
  export PBRS_SCALE="${PBRS_SCALE:-1.0}"
  export SEED
  export BEST_EVAL_SEED="${BEST_EVAL_SEED:-${SEED}}"
  export STUDY_ROOT
  export STUDY_FORCE_TRAIN
  export ALLOW_ARCHIVE_OVERWRITE
  export RUN_TAG="pub_${EXPERIMENT_SET}_${VARIANT_TAG}_n${N}_seed${SEED}"

  echo "Starting publication experiment"
  echo "experiment_set=${EXPERIMENT_SET}"
  echo "run_tag=${RUN_TAG}"
  echo "seed=${SEED}"
  echo "n=${N}"
  echo "budgets=${BUDGETS}"
  echo "eval_episodes=${EVAL_EPISODES}"
  echo "dueling=${DUELING} double_dqn=${DOUBLE_DQN} pbrs=${PBRS} pbrs_scale=${PBRS_SCALE}"

  bash scripts/run_stage2_study.sh
done
