import copy
import os

import torch

from helper_local import initialize_policy, get_config, DictToArgs, latest_model_path, create_venv, create_venv_render
from imitation_rl import decompose_policy


def get_env_args(cfg):
    # manual implementation for now
    env_args = {
        "val_env_name": cfg["val_env_name"],
        "env_name": cfg["env_name"],
        "num_levels": cfg["num_levels"],
        "start_level": cfg["start_level"],
        "distribution_mode": cfg["distribution_mode"],
        "num_threads": cfg["num_threads"],
        "random_percent": 50,#cfg["random_percent"],
        "step_penalty": cfg["step_penalty"],
        "key_penalty": cfg["key_penalty"],
        "rand_region": cfg["rand_region"],
        "param_name": cfg["param_name"],
    }
    return DictToArgs(env_args)




def watch_agent(logdir, next_val_dir):
    # load configs
    agent_dir = logdir
    cfg = get_config(agent_dir)
    args = get_env_args(cfg)
    hyperparameters = cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load env
    cfg["n_envs"] = 2
    venv = create_venv_render(args, cfg)
    # load agent
    model, policy = initialize_policy(device, hyperparameters, venv, venv.observation_space.shape)
    model.device = device
    policy.device = device
    # load policy
    last_model = latest_model_path(agent_dir)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])

    # load next value network:
    args_dict = get_config(next_val_dir,"args_dict.npy")
    level = args_dict.get("level")
    custom_embedder, next_value_network = decompose_policy(args_dict.get("new_val_weights"), device, level, model, policy)

    val_model = os.path.join(next_val_dir,"next_val_net.pth")
    next_value_network.load_state_dict(torch.load(val_model, map_location=device))

    # run agent in env
    obs = venv.reset()
    x = torch.FloatTensor(obs).to(device)
    x = custom_embedder(x)
    p, v, _ = policy(x, None, None)
    vn = next_value_network.value(x)
    while True:
        act = p.sample()
        obs, rew, done, info = venv.step(act.detach().cpu().numpy())
        predicted_reward = p.log_prob(act) + v - cfg["gamma"] * vn
        x = torch.FloatTensor(obs).to(device)
        x = custom_embedder(x)
        p, v, _ = policy(x, None, None)
        vn = next_value_network.value(x)
        # # pn, vn, _ = policy(x, None, None)
        # # vn[done] = 0
        # predicted_reward = p.log_prob(act) + v - cfg["gamma"] * vn
        # p = pn
        # v = vn
        print(f"Reward:{rew[0]:.2f}\tPredicted Reward:{predicted_reward[0]:.2f}\tValue:{v[0]:.2f}\tNV:{vn[0]:.2f}")

if __name__ == "__main__":
    #TODO: get agent_dir from config
    # have shifted explicit
    unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"
    logdir = shifted_agent_dir
    # unshifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-22__14-30-14__seed_6033"
    # shifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-22__14-57-56__seed_6033"
    next_val_dir = shifted_val_dir
    watch_agent(logdir=logdir, next_val_dir=next_val_dir)
