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




def watch_agent(logdir, next_val_dir=None):
    if next_val_dir is not None:
        args_dict = get_config(next_val_dir,"args_dict.npy")
        logdir = args_dict.get("agent_dir")
    args, cfg, device, model, policy, venv = load_policy(logdir)

    shenv = create_venv_render(args, cfg, is_valid=True)
    # for i in range(20):
    #     shp = policy(torch.FloatTensor(shenv.obs(i)[:1,]).to(device), None, None)[0].probs.round(decimals=2)
    #     p = policy(torch.FloatTensor(venv.obs(i)[:1,]).to(device), None, None)[0].probs.round(decimals=2)
    #     print(i)
    #     print(p)
    #     print(shp)

    # print_layers(policy.named_children(), x)

    for env in [venv, shenv]:
        x = torch.FloatTensor(env.obs(18)[:1,]).to(device)
        print(x.detach().cpu().numpy().round(decimals=2))
        print("Embedder Weight:")
        print(policy.embedder.layers[0].weight.detach().cpu().numpy().round(decimals=2))
        x = policy.embedder.layers[0](x)
        print("Latents:")
        print(x.detach().cpu().numpy().round(decimals=2))
        print("fc_policy Weight:")
        print(policy.fc_policy.weight.detach().cpu().numpy().round(decimals=2))
        print("outputs:")
        x = policy.fc_policy(x)
        print(x.detach().cpu().numpy().round(decimals=2))
        print("Probs:")
        print(x.softmax(dim=-1).detach().cpu().numpy().round(decimals=2))

    # load next value network:
    level = args_dict.get("level")
    custom_embedder, next_value_network = decompose_policy(args_dict.get("new_val_weights"), device, level, model, policy)

    val_model = os.path.join(next_val_dir,"next_val_net.pth")
    next_value_network.load_state_dict(torch.load(val_model, map_location=device))

    # run agent in env
    obs = venv.reset()
    while True:
        x = torch.FloatTensor(obs).to(device)
        x = custom_embedder(x)
        p, v, _ = policy(x, None, None)
        vn = next_value_network.value(x)
        act = p.sample()

        adv = p.log_prob(act)
        predicted_reward = adv + v - cfg["gamma"] * vn
        obs, rew, done, info = venv.step(act.detach().cpu().numpy())

        # These metrics are for the obs and action you just saw and took (not the state you can currently see rendered)
        print(f"Reward:{rew[0]:.2f}\tPredicted Reward:{predicted_reward[0]:.2f}\tAdv:{adv[0]:.2f}\tValue:{v[0]:.2f}\tNV:{vn[0]:.2f}\tEntropy:{p.entropy()[0]:.2f}")
        
        # x = torch.FloatTensor(obs).to(device)
        # x = custom_embedder(x)
        # p, v, _ = policy(x, None, None)
        # vn = next_value_network.value(x)


def load_policy(logdir, render=True, valid_env=False, n_envs=2):
    # load configs
    agent_dir = logdir
    cfg = get_config(agent_dir)
    args = get_env_args(cfg)
    hyperparameters = cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load env
    cfg["n_envs"] = n_envs
    if render:
        venv = create_venv_render(args, cfg, valid_env)
    else:
        venv = create_venv(args, cfg, valid_env)
    # load agent
    model, policy = initialize_policy(device, hyperparameters, venv, venv.observation_space.shape)
    model.device = device
    policy.device = device
    # load policy
    last_model = latest_model_path(agent_dir)
    policy.load_state_dict(torch.load(last_model, map_location=device)["model_state_dict"])
    return args, cfg, device, model, policy, venv


def print_layers(children, x):
    for name, layer in children:
        if name != "fc_value":
            print(x)
            print(name)
            if not hasattr(layer, "weight"):
                return print_layers(children, x)
            print(layer.weight.detach().cpu().numpy().round(decimals=2).tolist())
            x = layer(x)
    print(x)
    return x


if __name__ == "__main__":
    #TODO: get agent_dir from config
    # have shifted explicit
    unshifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    shifted_agent_dir = "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033"
    logdir = shifted_agent_dir
    # unshifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-22__14-30-14__seed_6033"
    # shifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-22__14-57-56__seed_6033"
    shifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-22__16-27-10__seed_6033"
    shifted_val_dir = "logs/next_val_finding/coinrun/coinrun/2024-10-24__10-19-48__seed_6033"
    next_val_dir = shifted_val_dir

    logdir = "logs/train/ascent/ascent/2024-11-12__13-52-38__seed_1080"
    watch_agent(logdir=logdir, next_val_dir=None)
