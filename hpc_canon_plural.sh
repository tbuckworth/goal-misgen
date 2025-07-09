#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=00:10:00
#PBS -N get_model_dirs

module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 run_canonicalisation_new.py --env_name $env_name --model_dir $model_dir
