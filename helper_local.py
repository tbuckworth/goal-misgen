import os
import random
import re

import gym
import gymnasium
import yaml
from procgen import ProcgenEnv

from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, TransposeFrame, ScaledFloatFrame, \
    DummyTerminalObsWrapper
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy


def create_venv(args, hyperparameters, is_valid=False):
    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    start_level_val = random.randint(0, 9999)
    venv = ProcgenEnv(num_envs=hyperparameters.get('n_envs', 256),
                      env_name=val_env_name if is_valid else args.env_name,
                      num_levels=0 if is_valid else args.num_levels,
                      start_level=start_level_val if is_valid else args.start_level,
                      distribution_mode=args.distribution_mode,
                      num_threads=args.num_threads,
                      random_percent=args.random_percent,
                      step_penalty=args.step_penalty,
                      key_penalty=args.key_penalty,
                      rand_region=args.rand_region)
    venv = VecExtractDictObs(venv, "rgb")
    normalize_rew = hyperparameters.get('normalize_rew', True)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False) # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    venv = DummyTerminalObsWrapper(venv)
    return venv

class DictToArgs:
    def __init__(self, input_dict):
        for key in input_dict.keys():
            setattr(self, key, input_dict[key])

def latest_model_path(logdir):
    logdir = os.path.join(logdir)
    files = os.listdir(logdir)
    pattern = r"model_(\d*)\.pth"
    checkpoints = [int(re.search(pattern, x).group(1)) for x in files if re.search(pattern, x)]
    last_model = os.path.join(logdir, f"model_{max(checkpoints)}.pth")
    return last_model

def initialize_policy(device, hyperparameters, env, observation_shape):
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space
    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete) or isinstance(action_space, gymnasium.spaces.discrete.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)
    return model, policy


def get_hyperparameters(param_name):
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    return hyperparameters
