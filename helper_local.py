import os
import random
import re

import gym
import gymnasium
import numpy as np
import yaml
from gym3 import ViewerWrapper, ToBaselinesVecEnv
from procgen import ProcgenGym3Env, ProcgenEnv

from common.ascent_env import AscentEnv
from common.env.procgen_wrappers import VecExtractDictObs, VecNormalize, TransposeFrame, ScaledFloatFrame, \
    DummyTerminalObsWrapper
from common.model import NatureModel, ImpalaModel, MlpModel
from common.policy import CategoricalPolicy


def create_venv(args, hyperparameters, is_valid=False):
    if args.env_name == "ascent":
        return AscentEnv(num_envs=hyperparameters.get('n_envs', 256),
                         shifted=is_valid,
                         num_positive_states=hyperparameters.get('n_pos_states', 20),
                         dense_rewards=hyperparameters.get('dense_rewards', False),
                         )

    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    # TODO: give this proper seed:
    #  also check if reset uses the same initial levels
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
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
    return venv


def create_venv_render(args, hyperparameters, is_valid=False):
    if args.env_name == "ascent":
        return AscentEnv(num_envs=hyperparameters.get('n_envs', 256),
                         shifted=is_valid,
                         num_positive_states=hyperparameters.get('n_states', 20),
                         )
    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    # TODO: give this proper seed:
    #  also check if reset uses the same initial levels
    start_level_val = random.randint(0, 9999)
    venv = ProcgenGym3Env(num=hyperparameters.get('n_envs', 256),
                          env_name=val_env_name if is_valid else args.env_name,
                          num_levels=0 if is_valid else args.num_levels,
                          start_level=start_level_val if is_valid else args.start_level,
                          distribution_mode=args.distribution_mode,
                          num_threads=args.num_threads,
                          random_percent=args.random_percent,
                          step_penalty=args.step_penalty,
                          key_penalty=args.key_penalty,
                          rand_region=args.rand_region,
                          render_mode="rgb_array",
                          )
    venv = ViewerWrapper(venv, info_key="rgb")
    venv = ToBaselinesVecEnv(venv)
    venv = VecExtractDictObs(venv, "rgb")
    normalize_rew = hyperparameters.get('normalize_rew', True)
    if normalize_rew:
        venv = VecNormalize(venv, ob=False)  # normalizing returns, but not
        # the img frames
    venv = TransposeFrame(venv)
    venv = ScaledFloatFrame(venv)
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
    elif architecture == 'mlpmodel':
        final_relu = hyperparameters.get('final_relu', False)
        hid_dims = hyperparameters.get('hid_dims', [3])
        model = MlpModel(input_dims=in_channels, hidden_dims=hid_dims, final_relu=final_relu)

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


def get_config(logdir, pathname="config.npy"):
    return np.load(os.path.join(logdir, pathname), allow_pickle='TRUE').item()


def listdir(path):
    return [os.path.join(path, d) for d in os.listdir(path)]