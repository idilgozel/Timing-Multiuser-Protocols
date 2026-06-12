#!/bin/bash -l
#$ -N qamel_pub_suite
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/myriad_logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ucapgoz/Timing-Multiuser-Protocols}"
LOG_DIR="${REPO_ROOT}/qamel/outputs/myriad_logs"
mkdir -p "${LOG_DIR}"

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source "${VENV_PATH:-/home/ucapgoz/venvs/timing-multiuser/bin/activate}"

cd "${REPO_ROOT}"

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

bash scripts/myriad_publication_suite.sh
