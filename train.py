import re

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.model import RewValModel, NextRewModel, MlpModelNoFinalRelu, ImpalaValueModel
from common.policy import UniformPolicy, CraftedTorchPolicy
from common.storage import Storage, LirlStorage
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
import random
import torch

from helper_local import create_venv, initialize_policy, get_hyperparameters, listdir, add_training_args, get_config, \
    create_shifted_venv, get_value_dir_and_config_for_env

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass


def train(args):
    exp_name = args.exp_name
    env_name = args.env_name
    start_level_val = random.randint(0, 9999)
    param_name = args.param_name
    gpu_device = args.gpu_device
    num_timesteps = int(args.num_timesteps)
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    if args.start_level == start_level_val:
        raise ValueError("Seeds for training and validation envs are equal.")

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    hyperparameters = get_hyperparameters(param_name)
    other_params = {}
    if args.model_file is not None:
        other_params = hyperparameters
        hyperparameters = get_config(re.sub(r"\/model_\d*.pth", "", args.model_file))
        if env_name == "get":
            args.env_name = env_name = hyperparameters.get("env_name")
        del hyperparameters["device"]
        del hyperparameters["num_checkpoints"]

    # override hyperparmeters:
    for var_name in list(hyperparameters.keys()) + list(other_params.keys()):
        if var_name in args.__dict__.keys() and args.__dict__[var_name] is not None:
            hyperparameters[var_name] = args.__dict__[var_name]

    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')

    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 256)
    algo = hyperparameters.get('algo', 'ppo')

    env = create_venv(args, hyperparameters)
    if algo not in ["trusted-value", "canon"]:
        env_valid = create_venv(args, hyperparameters, is_valid=True)
    else:
        env_valid = create_shifted_venv(args, hyperparameters)

    ############
    ## LOGGER ##
    ############

    def get_latest_model(model_dir):
        """given model_dir with files named model_n.pth where n is an integer,
        return the filename with largest n"""
        steps = [int(filename[6:-4]) for filename in os.listdir(model_dir) if filename.startswith("model_")]
        return list(os.listdir(model_dir))[np.argmax(steps)]

    print('INITIALIZING LOGGER...')

    logdir = create_logdir(args.model_file, env_name, exp_name, get_latest_model, listdir, seed)
    hyperparameters["logdir"] = logdir
    print(f'Logging to {logdir}')

    cfg = vars(args)
    cfg.update(hyperparameters)
    np.save(os.path.join(logdir, "config.npy"), cfg)

    if args.use_wandb:
        wandb_login()
        cfg = vars(args)
        cfg.update(hyperparameters)
        wb_resume = "allow"  # if args.model_file is None else "must"
        wandb.init(project="goal-misgen", config=cfg, tags=args.wandb_tags, resume=wb_resume)
    logger = Logger(n_envs, logdir, use_wandb=args.use_wandb)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape

    model, policy = initialize_policy(device, hyperparameters, env, observation_shape)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZING STORAGE...')
    storage, storage_valid, storage_trusted, storage_trusted_val = initialize_storage(device, model, n_envs, n_steps,
                                                                                      observation_shape, algo)

    if algo == 'ppo-lirl':
        hidden_dims = hyperparameters.get("hidden_dims", [64, 64])
        rew_epoch = hyperparameters.get("rew_epoch", 10)
        rew_lr = hyperparameters.get("rew_lr", 1e-5)
        action_size = env.action_space.n
        num_rew_updates = hyperparameters.get("num_rew_updates", 10)

        ppo_lirl_params = dict(
            num_rew_updates=num_rew_updates,
            rew_val_model=RewValModel(model.output_dim, hidden_dims, device),
            next_rew_model=NextRewModel(model.output_dim + action_size, hidden_dims, action_size, device),
            inv_temp_rew_model=1.,
            next_rew_loss_coef=1.,
            storage_trusted=storage_trusted,
            storage_trusted_val=storage_trusted_val,
            rew_epoch=rew_epoch,
            rew_lr=rew_lr,
        )
        hyperparameters.update(ppo_lirl_params)
    if algo in ['canon', 'trusted-value']:
        if hyperparameters.get("load_value_models", False):
            value_cfg, value_dir = get_value_dir_and_config_for_env(env_name, "Training")
            value_cfg_valid, value_dir_valid = get_value_dir_and_config_for_env(env_name, "Validation")
            hidden_dims = value_cfg.get("hidden_dims", [32])
        else:
            hidden_dims = hyperparameters.get("hidden_dims", [32])
        architecture = hyperparameters.get("architecture")
        if architecture == "impala":
            # This is a crazy amount of GPU, shall we think about doing something about all this?
            value_model = ImpalaValueModel(observation_shape[0], hidden_dims)
            value_model_val = ImpalaValueModel(observation_shape[0], hidden_dims)
            value_model_logp = ImpalaValueModel(observation_shape[0], hidden_dims)
            value_model_logp_val = ImpalaValueModel(observation_shape[0], hidden_dims)
            q_model = ImpalaValueModel(observation_shape[0], hidden_dims, output_dim=2)
            q_model_val = ImpalaValueModel(observation_shape[0], hidden_dims, output_dim=2)
        else:
            value_model = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [1])
            value_model_val = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [1])
            value_model_logp = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [1])
            value_model_logp_val = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [1])
            q_model = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [2])
            q_model_val = MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [2])

        value_model.to(device)
        value_model_val.to(device)
        value_model_logp.to(device)
        value_model_logp_val.to(device)
        q_model.to(device)
        q_model_val.to(device)

        trusted_policy_name = hyperparameters.get("trusted_policy", "uniform")
        if trusted_policy_name == "uniform":
            trusted_policy = UniformPolicy(policy.action_size, device, input_dims=len(env.observation_space.shape))
        elif trusted_policy_name == "gen":
            trusted_policy = CraftedTorchPolicy(False, policy.action_size, device,
                                                input_dims=len(env.observation_space.shape))
        elif trusted_policy_name == "misgen":
            trusted_policy = CraftedTorchPolicy(True, policy.action_size, device,
                                                input_dims=len(env.observation_space.shape))
        else:
            raise NotImplementedError

        canon_params = dict(
            value_model=value_model,
            value_model_val=value_model_val,
            storage_trusted=storage_trusted,
            storage_trusted_val=storage_trusted_val,
            trusted_policy=trusted_policy,
            value_model_logp=value_model_logp,
            value_model_logp_val=value_model_logp_val,
            q_model=q_model,
            q_model_val=q_model_val,
        )
        hyperparameters.update(canon_params)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    agent = initialize_agent(device, env, env_valid, hyperparameters, logger, num_checkpoints, policy, storage,
                             storage_valid)
    if args.model_file is not None:
        print("Loading agent from %s" % args.model_file)
        checkpoint = torch.load(args.model_file)
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if hyperparameters.get("load_value_models", False):
        checkpoint = torch.load(value_dir)
        agent.value_model.load_state_dict(checkpoint["model_state_dict"])
        agent.value_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint = torch.load(value_dir_valid)
        agent.value_model_val.load_state_dict(checkpoint["model_state_dict"])
        agent.value_optimizer_val.load_state_dict(checkpoint["optimizer_state_dict"])

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
    if args.use_wandb:
        wandb.finish()


def create_logdir(model_file, env_name, exp_name, get_latest_model, listdir, seed):
    logdir = os.path.join('logs', 'train', env_name, exp_name)
    if model_file == "auto":  # try to figure out which file to load
        logdirs_with_model = [d for d in listdir(logdir) if any(['model' in filename for filename in os.listdir(d)])]
        if len(logdirs_with_model) > 1:
            raise ValueError("Received args.model_file = 'auto', but there are multiple experiments"
                             f" with saved models under experiment_name {exp_name}.")
        elif len(logdirs_with_model) == 0:
            raise ValueError("Received args.model_file = 'auto', but there are"
                             f" no saved models under experiment_name {exp_name}.")
        model_dir = logdirs_with_model[0]
        model_file = os.path.join(model_dir, get_latest_model(model_dir))
        logdir = model_dir  # reuse logdir
    else:
        run_name = time.strftime("%Y-%m-%d__%H-%M-%S") + f'__seed_{seed}'
        logdir = os.path.join(logdir, run_name)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    return logdir


def initialize_storage(device, model, n_envs, n_steps, observation_shape, algo):
    hidden_state_dim = model.output_dim
    if algo == 'ppo':
        storage_cons = Storage
    elif algo in ['ppo-lirl', 'canon', 'trusted-value']:
        storage_cons = LirlStorage
    else:
        raise NotImplementedError(f"{algo} not implemented")
    storage = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_valid = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_trusted = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_trusted_val = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    return storage, storage_valid, storage_trusted, storage_trusted_val


def initialize_agent(device, env, env_valid, hyperparameters, logger, num_checkpoints, policy, storage, storage_valid):
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    elif algo == 'ppo-lirl':
        from agents.ppo_lirl import PPO_Lirl as AGENT
    elif algo == 'canon':
        from agents.canonicalise import Canonicaliser as AGENT
    elif algo == 'trusted-value':
        from agents.trusted_value import TrustedValue as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device,
                  num_checkpoints,
                  env_valid=env_valid,
                  storage_valid=storage_valid,
                  **hyperparameters)
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_training_args(parser)
    args = parser.parse_args()

    train(args)
