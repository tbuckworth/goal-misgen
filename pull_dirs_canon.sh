#!/bin/bash
#PBS -l select=1:ncpus=8:mem=4gb
#PBS -l walltime=00:10:00
#PBS -N get_model_dirs

module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 pull_dirs4canon.py --env_name $env_name
