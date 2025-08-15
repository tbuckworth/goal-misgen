#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N canon

cd /vol/bitbucket/${USER}/goal-misgen
source venv/bin/activate
python3.9 multi_train.py $1