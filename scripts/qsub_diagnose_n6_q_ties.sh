#!/bin/bash -l
#$ -N qamel_n6_qties
#$ -l h_rt=02:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -wd /home/ucapgoz/Timing-Multiuser-Protocols
#$ -j y
#$ -o /home/ucapgoz/Timing-Multiuser-Protocols/qamel/outputs/n6_q_ties.$JOB_ID.out

set -euo pipefail

module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/2.1.0/gpu
source /home/ucapgoz/venvs/timing-multiuser/bin/activate

cd /home/ucapgoz/Timing-Multiuser-Protocols

echo "commit=$(git rev-parse HEAD)"
python scripts/diagnose_q_tie_stalls.py \
  --run-name dqn_n6_pgen0.4_pswap0.7_stage0_fullstack_n6_v2_s202 \
  --seed 12345 \
  --episodes 200 \
  --max-actions 150 \
  --device cuda
