import argparse
import os

import numpy as np

from helper_local import get_rew_term, get_seed, get_model_with_largest_checkpoint, get_rew_sufficient
from hyperparameter_optimization import run_next_hyperparameters
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass


def get_performing_model_dirs(env_name):
    target_rew = get_rew_sufficient(env_name)
    wandb_login()
    api = wandb.Api()
    entity, project = "ic-ai-safety", "goal-misgen"

    filters = {
        "state": "finished",
        "config.algo": "ppo",
        "config.env_name": env_name,
        "summary_metrics.mean_episode_rewards": {"$gte": target_rew},
        "tags": {
            "$in": ["rbm_gen_2"],
        },
    }

    runs = api.runs(f"{entity}/{project}", filters=filters)
    model_dirs = []
    for run in runs:
        logdir = run.config.get("logdir")
        if logdir and os.path.exists(logdir):
            model_dirs.append(logdir)

    if not model_dirs:
        raise FileNotFoundError("No local models found for this env. Could have been trained elsewhere.")
    return model_dirs


steps = {
    "mountain_car": {
        "num_timesteps": int(40 * 512),
        "n_envs": 32,
        "n_steps": 512,
        "n_val_envs": 8,
        "val_epoch": 1000,  # todo was 150
        "hidden_dims": [256, 256],
    },
    "cartpole": {
        "num_timesteps": 2e6, #int(40 * 512),
        "n_envs": 32,
        "n_steps": 512,
        "n_val_envs": 8,
        "val_epoch": 100,  # todo was 150
        "hidden_dims": [256, 256],
    },
    "acrobot": {
        "num_timesteps": int(40 * 512),
        "n_envs": 32,
        "n_steps": 512,
        "n_val_envs": 8,
        "val_epoch": 100,  # todo was 150
        "hidden_dims": [256, 256],
    },
    "maze": {
        "num_timesteps": int(24 * 2048),
        "n_envs": 16,
        "n_steps": 2048,
        "n_val_envs": 8,
        "val_epoch": 300,
        "hidden_dims": [256, 256, 256, 256],
    },
    "maze_aisc": {
        "num_timesteps": int(24 * 2048),
        "n_envs": 16,
        "n_steps": 2048,
        "n_val_envs": 8,
        "val_epoch": 300,
        "hidden_dims": [256, 256, 256, 256],
    },
    "coinrun": {
        "num_timesteps": int(24 * 2048),
        "n_envs": 16,
        "n_steps": 2048,
        "n_val_envs": 8,
        "val_epoch": 300,
        "hidden_dims": [256, 256, 256, 256],
    },
    "ascent": {
        "num_timesteps": 2e6, #int(24 * 256),
        "n_envs": 16,
        "n_steps": 256,
        "n_val_envs": 8,
        "val_epoch": 100,
        "hidden_dims": [16, 16],
    },
}

configs = {
    "soft_inf": {
        "soft_canonicalisation": True,
        "infinite_value": True,
        "centered_logprobs": False,
        "adjust_logprob_mean": False,
    },
    "soft_no_inf_mean_adj": {
        "soft_canonicalisation": True,
        "infinite_value": False,
        "centered_logprobs": False,
        "adjust_logprob_mean": True,
    },
    # "soft_no_inf": {
    #     "soft_canonicalisation": True,
    #     "infinite_value": False,
    #     "centered_logprobs": False,
    #     "adjust_logprob_mean": False,
    # },
    # "hard": {
    #     "soft_canonicalisation": False,
    #     "infinite_value": False,
    #     "centered_logprobs": False,
    #     "adjust_logprob_mean": False,
    # },
    # "hard_centred": {
    #     "soft_canonicalisation": False,
    #     "infinite_value": False,
    #     "centered_logprobs": True,
    #     "adjust_logprob_mean": False,
    # },
    # "hard_mean_adj": {
    #     "soft_canonicalisation": False,
    #     "infinite_value": False,
    #     "centered_logprobs": False,
    #     "adjust_logprob_mean": True,
    # },
}


def run_canonicalisation(model_file, env_name, config, suffix):
    seed = get_seed(model_file)
    step_dict = steps.get(env_name)
    config_dict = configs.get(config)

    hparams = {
        "model_file": get_model_with_largest_checkpoint(model_file),
        "epoch": 0,
        "algo": "canon",
        "env_name": "get",
        "exp_name": "canon",
        "param_name": "cartpole-canon",
        "wandb_tags": [f"{env_name}_{config}_{suffix}", "rbm_canon_8"],
        "num_checkpoints": 1,
        "use_wandb": True,
        "mini_batch_size": 2048,
        "num_levels": 10000,
        "learning_rate": 5e-4,
        "distribution_mode": "hard-slow-learn",
        "seed": seed,
        "load_value_models": False,  # todo check if value functions are good
        # "pre_trained_value_encoder": True,
        "value_dir": None,
        # "soft_canonicalisation": True,
        "update_frequently": True,
        # "infinite_value": True,
        "meg": False,
        "remove_duplicate_actions": True,
        # "centered_logprobs": False,
        # "adjust_logprob_mean": False,
        "use_valid_env": True,
        "pirc": True,
        "trusted_policy": "uniform",  # todo can potentially use trusted policy
        "trusted_temp": 7,
        "use_min_val_loss": False,
    }
    hparams.update(step_dict)
    hparams.update(config_dict)
    run_next_hyperparameters(hparams)


# def main(env_name):
#     model_dirs = get_performing_model_dirs(env_name)
#     suffix = np.random.randint(0, 10000)
#     for model_file in model_dirs:
#         for config in configs.keys():
#             run_canonicalisation(model_file, env_name, config, suffix)



def _worker(args):
    """Thin wrapper to make the executor happy."""
    model_file, config, env_name, suffix = args
    run_canonicalisation(model_file, env_name, config, suffix)

def main(env_name: str, num_workers) -> None:
    """
    Fire off run_canonicalisation in parallel.

    Parameters
    ----------
    env_name : str
        Environment name passed straight through.
    num_workers : int | None, optional
        How many worker processes to launch.  None → `os.cpu_count()`.
    """
    model_dirs = get_performing_model_dirs(env_name)
    suffix = np.random.randint(0, 10_000)

    # Build the argument tuples once; avoids re-rolling suffix in each process.
    jobs = [
        (model_file, config, env_name, suffix)
        for model_file in model_dirs
        for config in configs.keys()
    ]
    
    # Spawn is the safe option with PyTorch.
    
    # mp.set_start_method("spawn", force=True)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        pool.map(_worker, jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="ascent")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    # main("ascent", 1)
    main(args.env_name, args.workers)
