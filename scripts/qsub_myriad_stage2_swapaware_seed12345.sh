#!/bin/bash -l
#$ -N qnet_swapaware
#$ -l h_rt=12:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/cluster_logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

mkdir -p /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/cluster_logs

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source /home/ucapgoz/venvs/timing-multiuser/bin/activate

cd /home/ucapgoz/Timing-Multiuser-Protocols

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

bash scripts/myriad_stage2_swapaware_seed12345.sh
