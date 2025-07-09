#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N canon

module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 procgen_canon_fixed.py --env_name coinrun

