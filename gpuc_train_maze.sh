#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=canon

module load CMake/3.18.4-GCCcore-10.2.0
module load Qt5/5.15.2-GCCcore-10.3.0
module load Python/3.8.6-GCCcore-10.2.0

# Set Qt5 directory for CMake
export Qt5_DIR=$EBROOTQT5/lib/cmake/Qt5

# TODO hardcoded my username since slurm does not seem to have USER env var
cd /vol/bitbucket/rm1723/goal-misgen/
# source /vol/bitbucket/tfb115/goal-misgen/venv/bin/activate
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ../procgenAISC/
python3.8 train.py --exp_name maze_aisc --env_name maze_aisc --num_levels $num_levels --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed $seed --use_wandb --rand_region $rand_region

#usage:
# sbatch --export=rand_region=10,num_levels=100000,seed=42 hpc_train_maze.sh
