from hyperparameter_optimization import run_next_hyperparameters


if __name__ == '__main__':
    hparams = {
        # "num_levels": 100000,
        "architecture": "mlpmodel",
        # "distribution_mode": "hard",
        "seed": 1080,
        "env_name": "ascent",
        "exp_name": "value",
        "param_name": "trusted-value",
        "wandb_tags": ["trusted value"],
        "use_wandb": True,
        "val_epoch": 2000,
        "mini_batch_size": 2048,
        "n_val_envs": 128,
        "n_envs": 256 + 128,
    }
    run_next_hyperparameters(hparams)
