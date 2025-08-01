#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rm1723

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib
source /vol/bitbucket/${USER}/goal-misgen/venv/bin/activate
. /vol/cuda/12.2.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/
python3.9 /vol/bitbucket/${USER}/goal-misgen/canon_clean.py --env_name $1 --workers $2
