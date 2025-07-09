module load Python/3.8.6-GCCcore-10.2.0
cd $HOME/pyg/goal-misgen/
source venv/bin/activate
python3.8 pull_dirs4canon.py --env_name $1
