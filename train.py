from common.env.procgen_wrappers import *
from common.logger import Logger
from common.model import RewValModel, NextRewModel
from common.storage import Storage, LirlStorage
from common import set_global_seeds, set_global_log_levels

import os, time, argparse
import random
import torch

from helper_local import create_venv, initialize_policy, get_hyperparameters, listdir

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass



def train(args):
    exp_name = args.exp_name
    env_name = args.env_name
    val_env_name = args.val_env_name if args.val_env_name else args.env_name
    start_level = args.start_level
    start_level_val = random.randint(0, 9999)
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
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
    env_valid = create_venv(args, hyperparameters, is_valid=True)


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

    print(f'Logging to {logdir}')

    cfg = vars(args)
    cfg.update(hyperparameters)
    np.save(os.path.join(logdir, "config.npy"), cfg)

    if args.use_wandb:
        wandb_login()
        cfg = vars(args)
        cfg.update(hyperparameters)
        wb_resume = "allow" if args.model_file is None else "must"
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
    storage, storage_valid, storage_trusted = initialize_storage(device, model, n_envs, n_steps, observation_shape, algo)

    if algo == 'ppo-lirl':
        hidden_dims = hyperparameters.get("hidden_dims", [64, 64])
        action_size = env.action_space.n

        ppo_lirl_params = dict(
            num_rew_updates=10,
            rew_val_model=RewValModel(model.output_dim, hidden_dims, device),
            next_rew_model=NextRewModel(model.output_dim + action_size, hidden_dims, action_size, device),
            inv_temp_rew_model=1.,
            next_rew_loss_coef=1.,
            storage_trusted=storage_trusted,
        )
        hyperparameters.update(ppo_lirl_params)


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

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)


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
    elif algo == 'ppo-lirl':
        storage_cons = LirlStorage
    else:
        raise NotImplementedError (f"{algo} not implemented")
    storage = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_valid = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    storage_trusted = storage_cons(observation_shape, hidden_state_dim, n_steps, n_envs, device)
    return storage, storage_valid, storage_trusted


def initialize_agent(device, env, env_valid, hyperparameters, logger, num_checkpoints, policy, storage, storage_valid):
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    if algo == 'ppo-lirl':
        from agents.ppo_lirl import PPO_Lirl as AGENT
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
    args = parser.parse_args()

    train(args)