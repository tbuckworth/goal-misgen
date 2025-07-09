qsub -v random_percent=0,seed=6033,num_levels=1000 hpc_train.sh
qsub -v random_percent=0,seed=6033,num_levels=100000 hpc_train.sh
qsub -v random_percent=0,seed=42,num_levels=1000 hpc_train.sh
qsub -v random_percent=0,seed=42,num_levels=100000 hpc_train.sh
qsub -v random_percent=10,seed=6033,num_levels=1000 hpc_train.sh
qsub -v random_percent=10,seed=6033,num_levels=100000 hpc_train.sh
qsub -v random_percent=10,seed=42,num_levels=1000 hpc_train.sh
qsub -v random_percent=10,seed=42,num_levels=100000 hpc_train.sh
