import torch

from helper_local import initialize_policy, get_config, DictToArgs, latest_model_path, create_venv, create_venv_render

def get_env_args(cfg):
    # manual implementation for now
    env_args = {
        "val_env_name": cfg["val_env_name"],
        "env_name": cfg["env_name"],
        "num_levels": cfg["num_levels"],
        "start_level": cfg["start_level"],
        "distribution_mode": cfg["distribution_mode"],
        "num_threads": cfg["num_threads"],
        "random_percent": 100,#cfg["random_percent"],
        "step_penalty": cfg["step_penalty"],
        "key_penalty": cfg["key_penalty"],
        "rand_region": cfg["rand_region"],
        "param_name": cfg["param_name"],
    }
    return DictToArgs(env_args)




def watch_agent(logdir):
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
    # run agent in env
    obs = venv.reset()
    x = torch.FloatTensor(obs).to(device)
    p, v, _ = policy(x, None, None)
    while True:
        act = p.sample()
        obs, rew, done, info = venv.step(act.detach().cpu().numpy())
        x = torch.FloatTensor(obs).to(device)
        pn, vn, _ = policy(x, None, None)
        vn[done] = 0
        predicted_reward = p.log_prob(act) + v - cfg["gamma"] * vn
        p = pn
        v = vn
        print(f"Reward:{rew[0]}\tPredicted Reward:{predicted_reward[0]:.2f}\tValue:{v[0]:.2f}")

if __name__ == "__main__":
    logdir = "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033"
    watch_agent(logdir=logdir)