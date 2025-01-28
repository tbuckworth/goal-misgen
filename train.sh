#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tfb115

export PATH=/vol/bitbucket/${USER}/goal-misgen/opvenv/bin/:/vol/cuda/12.2.0/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib
source /vol/bitbucket/${USER}/goal-misgen/opvenv/bin/activate
. /vol/cuda/12.2.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/
#python3.8 /vol/bitbucket/${USER}/goal-misgen/train.py --exp_name coinrun --env_name coinrun --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 6033 --use_wandb
#python3.8 /vol/bitbucket/${USER}/goal-misgen/train.py --exp_name maze1 --env_name maze_aisc --num_levels 500 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 42 --use_wandb
python3.8 /vol/bitbucket/${USER}/goal-misgen/train.py --exp_name cartpole --env_name cartpole --param_name cartpole-mlp --num_timesteps 10000000 --num_checkpoints 1 --seed 6033 --use_wandb
