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
from common.policy import CategoricalPolicy, CraftedTorchPolicy


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
    elif architecture == 'crafted-policy':
        misgen = hyperparameters.get('misgen', False)
        action_size = action_space.n
        policy = CraftedTorchPolicy(misgen, action_size, device)
        model = DictToArgs({"output_dim": 3})
        return model, policy

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


def add_training_args(parser):
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='gpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--wandb_tags', type=str, nargs='+')
    parser.add_argument('--random_percent', type=int, default=0,
                        help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0,
                        help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0,
                        help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0,
                        help='MAZE: size of region (in upper left corner) in which goal is sampled.')
    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)
    return parser


def get_model_with_largest_checkpoint(folder):
    files = [os.path.join(folder, x) for x in os.listdir(folder)]
    search = lambda x: re.search(r"model_(\d*).pth", x)
    last_checkpoint = max([int(search(x).group(1)) for x in files if search(x)])
    return [x for x in files if re.search(f"model_{last_checkpoint}.pth", x)][0]
