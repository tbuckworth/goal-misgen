import argparse

import numpy as np

from canon_clean import steps, configs, get_performing_model_dirs
from helper_local import get_model_with_largest_checkpoint, get_config
from hyperparameter_optimization import run_next_hyperparameters
from load_wandb_table import create_from_tag


def run_canonicalisation(model_file, env_name, config, suffix):
    step_dict = steps.get(env_name)
    config_dict = configs.get(config)

    orig_cfg = get_config(model_file)

    required_cols = ['seed', 'distribution_mode', 'num_levels',  'start_level']
    # 'param_name','rand_region', 'random_percent',?
    orig_dict = {k:v for k,v in orig_cfg.items() if k in required_cols}

    hparams = {
        "model_file": get_model_with_largest_checkpoint(model_file),
        "epoch": 0,
        "algo": "canon",
        "env_name": "get",
        "exp_name": "canon",
        "param_name": "cartpole-canon",
        "wandb_tags": [f"{env_name}_{config}_{suffix}"],
        "num_checkpoints": 1,
        "use_wandb": True,
        "mini_batch_size": 2048,
        "learning_rate": 5e-4,
        "load_value_models": False,
        # "pre_trained_value_encoder": True,
        "value_dir": None,
        "soft_canonicalisation": True,
        "update_frequently": True,
        "infinite_value": True,
        "meg": False,
        "remove_duplicate_actions": True,
        "centered_logprobs": False,
        "adjust_logprob_mean": False,
        "use_valid_env": True,
        "pirc": True,
        "trusted_policy": "uniform",
        "trusted_temp": 7,
        "use_min_val_loss": False,
    }
    hparams.update(step_dict)
    hparams.update(config_dict)
    hparams.update(orig_dict)
    run_next_hyperparameters(hparams)

def main(env_name, config):
    model_dirs = get_performing_model_dirs(env_name)
    suffix = np.random.randint(0, 10_000)

    for model_file in model_dirs:
        run_canonicalisation(model_file, env_name, config, suffix)

    tag = f"{env_name}_{config}_{suffix}"
    create_from_tag(tag, title = f"{env_name} {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="maze_aisc")
    parser.add_argument("--config", type=str, default="soft_inf")
    args = parser.parse_args()
    main(args.env_name, args.config)
