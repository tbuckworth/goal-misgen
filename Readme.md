# Setup Instructions on Imperial lab machines

```
curl https://pyenv.run | bash
```

Then you should see:

```
WARNING: seems you still have not added 'pyenv' to the load path.

Load pyenv automatically by adding
the following to ~/.bashrc:

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
 
Once you've done that and possibly restarted the shell:

```
pyenv install -v 3.8.0
pyenv local 3.8.0
git clone https://github.com/tbuckworth/goal-misgen.git
```

To create the virtual environment:

```
virtualenv -p python3.8 path_to_your_venv
source path_to_your_venv/bin/activate
pip install gym gym3 matplotlib numpy pandas Pillow PyYAML seaborn torch torchvision tqdm imitation wandb
```

Installing the modified ProcGen:

```
git clone https://github.com/JacobPfau/procgenAISC.git
cd procgenAISC
pip install -e .
```

test it out:

```
cd -
python train.py --exp_name coinrun --env_name coinrun --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 6033 --random_percent 0
```

maze variant 1 - misgeneralizing:
```
python train.py --exp_name maze1 --env_name maze_aisc --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 1080 --rand_region 0 --use_wandb
```
maze variant1 - generalizing:
```
python train.py --exp_name maze1 --env_name maze_aisc --num_levels 100000 --distribution_mode hard --param_name hard-500 --num_timesteps 200000000 --num_checkpoints 5 --seed 1080 --rand_region 16 --use_wandb
```


# Matt Notes

My env is in /vol/bitbucket/mjm121/goal-misgenv

Had to use the flag --use-pep517 when doing the pip install

Easiest way to work on the project is to shell into a labmachine (e.g. beech12) within VScode or cursor
```
python3.8 train.py --exp_name ascent --env_name ascent --param_name ascent-mlp --num_timesteps 2000000 --num_checkpoints 1 --seed 1080 --use_wandb --use_valid_env
```