import copy

from common.env.procgen_wrappers import *
from common.logger import Logger
from common.model import RewValModel, NextRewModel, MlpModelNoFinalRelu, ImpalaValueModel
from common.policy import UniformPolicy, CraftedTorchPolicy, ValuePolicyWrapper
from common.storage import Storage, LirlStorage
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
import random
import torch

from discrete_env.stacked_env import StackedEnv
from helper_local import create_venv, initialize_policy, get_hyperparameters, listdir, add_training_args, get_config, \
    create_shifted_venv, get_value_dir_and_config_for_env, create_unshifted_venv, get_model_with_largest_checkpoint, \
    load_tempered_policy

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass


def load_ppo_value_models(env_name, env, observation_shape, device, val_type):
    if env_name == "maze" or env_name == "maze_aisc":
        logdir = "logs/train/maze_aisc/value/2025-01-23__custom__seed_42"
    else:
        raise NotImplementedError("ppo value models only trained for maze so far")
    value_dir = os.path.join(logdir, val_type)
    cfg = get_config(value_dir)
    _, policy = initialize_policy(device, cfg, env, observation_shape)
    return policy, get_model_with_largest_checkpoint(value_dir)


def get_rew_term(env_name):
    if env_name == "cartpole" or env_name == "cartpole_swing":
        return 495
    if env_name == "mountain_car":
        return -100
    if env_name == "acrobot":
        return -100
    if env_name == "coinrun":
        return 9.9
    if env_name == "maze" or env_name == "maze_aisc":
        return 9.9
    if env_name == "ascent":
        return 9.9
    else:
        raise NotImplementedError(f"reward termination not implemented for {env_name}")


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

    if algo not in ["trusted-value", "canon"]:
        pct_ood = hyperparameters.get("train_pct_ood", 0)
        if pct_ood > 0:
            env = create_stacked_env(args, hyperparameters, pct_ood)
        else:
            env = create_venv(args, hyperparameters)
        env_valid = create_venv(args, hyperparameters, is_valid=True) if args.use_valid_env else None
    else:
        env = create_unshifted_venv(args, hyperparameters)
        env_valid = create_shifted_venv(args, hyperparameters) if args.use_valid_env else None

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

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    act_shape = (env.action_space.n,)

    model, policy = initialize_policy(device, hyperparameters, env, observation_shape)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZING STORAGE...')
    pre_trained_value_encoder = hyperparameters.get("pre_trained_value_encoder", False)
    is_impala = hyperparameters.get("architecture", "impala")=="impala"
    storage_override = (256,) if (pre_trained_value_encoder and is_impala) else None
    storage, storage_valid, storage_trusted, storage_trusted_val = initialize_storage(device, model, n_envs, n_steps,
                                                                                      observation_shape, algo, act_shape, storage_override)

    ppo_value = True if hyperparameters.get("value_dir", None) == "ppo" else False
    if ppo_value:
        value_model, value_dir = load_ppo_value_models(env_name, env, observation_shape, device, "Training")
        value_model_val, value_dir_valid = load_ppo_value_models(env_name, env, observation_shape, device, "Validation")

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
    canon_params = {}
    if algo in ['canon', 'trusted-value', 'trusted-value-unlimited']:
        trusted_policy_name = hyperparameters.get("trusted_policy", "uniform")
        if not ppo_value:
            if hyperparameters.get("load_value_models", False):
                valdir = hyperparameters.get("value_dir", None)
                value_cfg, value_dir = get_value_dir_and_config_for_env(env_name, "Training", valdir, trusted_policy_name)
                value_cfg_valid, value_dir_valid = get_value_dir_and_config_for_env(env_name, "Validation", valdir, trusted_policy_name)
                hidden_dims = value_cfg.get("hidden_dims", [32])
                if valdir is None:
                    hyperparameters["value_dir"] = value_dir
            else:
                hidden_dims = hyperparameters.get("hidden_dims", [32])
            model_constructor, value_model, value_model_val = construct_value_models(device, hyperparameters,
                                                                                     observation_shape, hidden_dims)
        else:
            hidden_dims = hyperparameters.get("hidden_dims", [32])
            model_constructor = get_value_constructor(hyperparameters, observation_shape, hidden_dims)
        # if hyperparameters.get("soft_canonicalisation", False):
        value_model_logp = model_constructor(1)
        value_model_logp_val = model_constructor(1)
        value_model_logp.to(device)
        value_model_logp_val.to(device)

        if hyperparameters.get("meg", False):
            output_dim = env.action_space.n
            if hyperparameters.get("meg_version", None) == "kldiv":
                output_dim = [output_dim, 1]
            q_model = model_constructor(output_dim)
            q_model_val = model_constructor(output_dim)
            q_model.to(device)
            q_model_val.to(device)
        else:
            q_model = q_model_val = None

        if trusted_policy_name == "uniform":
            trusted_policy = UniformPolicy(policy.action_size, device, input_dims=len(env.observation_space.shape))
        elif trusted_policy_name == "gen":
            trusted_policy = CraftedTorchPolicy(False, policy.action_size, device,
                                                input_dims=len(env.observation_space.shape))
        elif trusted_policy_name == "misgen":
            trusted_policy = CraftedTorchPolicy(True, policy.action_size, device,
                                                input_dims=len(env.observation_space.shape))
        elif trusted_policy_name == "self":
            trusted_policy = policy
        elif trusted_policy_name == "tempered_gen":
            trusted_policy = load_tempered_policy(env_name, device, hyperparameters, env)
            trusted_policy.T = hyperparameters.get("trusted_temp", 5)
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

    hyperparameters.update(canon_params)

    rew_term = hyperparameters.get("reward_termination", None)
    if rew_term is not None:
        if isinstance(rew_term, str) and rew_term=="get":
            hyperparameters["reward_termination"] = get_rew_term(env_name)


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
        checkpoint_val = torch.load(value_dir_valid)
        agent.value_model_val.load_state_dict(checkpoint_val["model_state_dict"])
        agent.value_optimizer_val.load_state_dict(checkpoint_val["optimizer_state_dict"])
        if hyperparameters.get("pre_trained_value_encoder", False):
            if isinstance(agent.value_model, ImpalaValueModel):
                agent.encoder = agent.value_model.model
                agent.encoder_val = agent.value_model_val.model

                # agent.value_model_logp.load_state_dict(checkpoint["model_state_dict"])
                # agent.value_model_logp.fix_encoder_reset_rest(agent.value_optimizer_logp)
                # agent.value_model_logp_val.load_state_dict(checkpoint_val["model_state_dict"])
                # agent.value_model_logp_val.fix_encoder_reset_rest(agent.value_optimizer_logp_val)
        if ppo_value:
            agent.value_model = ValuePolicyWrapper(agent.value_model)
            agent.value_model_val = ValuePolicyWrapper(agent.value_model_val)
    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
    if args.use_wandb:
        wandb.finish()


def create_stacked_env(args, hyperparameters, pct_ood):
    n_envs = hyperparameters.get("n_envs", 256)
    n_ood_envs = int(round(pct_ood * n_envs, 0))
    n_reg_envs = n_envs - n_ood_envs
    ood_hp = copy.deepcopy(hyperparameters)
    reg_hp = copy.deepcopy(hyperparameters)
    ood_hp["n_envs"] = n_ood_envs
    reg_hp["n_envs"] = n_reg_envs
    reg_env = create_venv(args, reg_hp)
    ood_env = create_venv(args, ood_hp, is_valid=True)
    env = StackedEnv([reg_env, ood_env])
    return env


def construct_value_models(device, hyperparameters, observation_shape, hidden_dims):
    model_constructor = get_value_constructor(hyperparameters, observation_shape, hidden_dims, False)
    # This is a crazy amount of GPU, shall we think about doing something about all this?
    value_model = model_constructor(1)
    value_model_val = model_constructor(1)
    value_model.to(device)
    value_model_val.to(device)
    model_constructor = get_value_constructor(hyperparameters, observation_shape, hidden_dims, True)
    return model_constructor, value_model, value_model_val


def get_value_constructor(hyperparameters, observation_shape, hidden_dims, use_encoder=False):
    architecture = hyperparameters.get("architecture")
    if use_encoder:
        use_encoder = hyperparameters.get("pre_trained_value_encoder", False)
    if architecture == "impala":
        if not use_encoder:
            return lambda x: ImpalaValueModel(observation_shape[0], hidden_dims, x)
        # 256 is the output_dim of an ImpalaModel
        return lambda x: MlpModelNoFinalRelu(256, hidden_dims + [x])
    return lambda x: MlpModelNoFinalRelu(observation_shape[0], hidden_dims + [x])


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


def initialize_storage(device, model, n_envs, n_steps, observation_shape, algo, act_shape, storage_override=None):
    hidden_state_dim = model.output_dim
    storage_cons = Storage
    if algo in ['ppo-lirl', 'canon', 'trusted-value', 'trusted-value-unlimited', 'ppo-tracked']:
        storage_cons = LirlStorage
    storage = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device, act_shape)
    storage_valid = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device, act_shape)
    hidden_state_dim = (storage_override or hidden_state_dim)
    storage_trusted = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device, act_shape)
    storage_trusted_val = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device, act_shape)

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
    elif algo == 'trusted-value-unlimited':
        from agents.trusted_value_unlimited import TrustedValue as AGENT
    elif algo == 'bpo':
        from agents.bpo import BPO as AGENT
    elif algo == 'ppo-uniform':
        from agents.ppo_uniform import PPO_Uniform as AGENT
    elif algo == 'ppo-tracked':
        from agents.ppo_tracked import PPO_Tracked as AGENT
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
