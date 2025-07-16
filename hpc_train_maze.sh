#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N canon

module load CMake/3.18.4-GCCcore-10.2.0
module load Qt5/5.15.2-GCCcore-10.3.0
module load Python/3.8.6-GCCcore-10.2.0

# Set Qt5 directory for CMake
export Qt5_DIR=$EBROOTQT5/lib/cmake/Qt5

cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ../procgenAISC/
python3.8 train.py --exp_name maze_aisc --env_name maze_aisc --num_levels $num_levels --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed $seed --use_wandb --rand_region $rand_region

#usage:
# qsub -v rand_region=10,num_levels=100000,seed=42 hpc_train_maze.sh
