#!/bin/bash

# Define your lists
rand_regions=(0 10)
seeds=(6033 42 902 1034)
num_levels_list=(1000 100000)

# Nested loops to generate all combinations
for rand_region in "${rand_regions[@]}"; do
    for seed in "${seeds[@]}"; do
        for num_levels in "${num_levels_list[@]}"; do
            qsub -v rand_region=${rand_region},seed=${seed},num_levels=${num_levels} hpc_train_maze.sh
        done
    done
done
