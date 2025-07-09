#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N canon

module load CMake/3.18.4-GCCcore-10.2.0





module load Qt5/5.15.2-GCCcore-10.3.0










module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 train.py --exp_name coinrun --env_name coinrun --num_levels $num_levels --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed $seed --use_wandb --random_percent $random_percent

#usage:
# qsub -v random_percent=10,num_levels=100000,seed=42 hpc_train.sh
