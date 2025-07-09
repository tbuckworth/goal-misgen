#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N get_model_dirs

module load CMake/3.18.4-GCCcore-10.2.0
module load Qt5/5.15.2-GCCcore-10.3.0
module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 run_canonicalisation_new.py --env_name $env_name --model_dir $model_dir --config $config --suffix $suffix
