qsub -v rand_region=0,seed=6033,num_levels=1000 hpc_train_maze.sh
qsub -v rand_region=0,seed=6033,num_levels=100000 hpc_train_maze.sh
qsub -v rand_region=0,seed=42,num_levels=1000 hpc_train_maze.sh
qsub -v rand_region=0,seed=42,num_levels=100000 hpc_train_maze.sh
qsub -v rand_region=10,seed=6033,num_levels=1000 hpc_train_maze.sh
qsub -v rand_region=10,seed=6033,num_levels=100000 hpc_train_maze.sh
qsub -v rand_region=10,seed=42,num_levels=1000 hpc_train_maze.sh
qsub -v rand_region=10,seed=42,num_levels=100000 hpc_train_maze.sh
