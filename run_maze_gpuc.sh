sbatch --export=rand_region=0,seed=6033,num_levels=1000 gpuc_train_maze.sh
sbatch --export=rand_region=0,seed=6033,num_levels=100000 gpuc_train_maze.sh
sbatch --export=rand_region=0,seed=42,num_levels=1000 gpuc_train_maze.sh
sbatch --export=rand_region=0,seed=42,num_levels=100000 gpuc_train_maze.sh
sbatch --partition training --export=rand_region=10,seed=6033,num_levels=1000 gpuc_train_maze.sh
sbatch --partition training --export=rand_region=10,seed=6033,num_levels=100000 gpuc_train_maze.sh
sbatch --partition training --export=rand_region=10,seed=42,num_levels=1000 gpuc_train_maze.sh
sbatch --partition training --export=rand_region=10,seed=42,num_levels=100000 gpuc_train_maze.sh