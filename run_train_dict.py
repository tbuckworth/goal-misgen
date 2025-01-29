from hyperparameter_optimization import run_next_hyperparameters

if __name__ == '__main__':
    hparams = {
        # "num_levels": 100000,
        "architecture": "mlpmodel",
        # "distribution_mode": "hard",
        "seed": 1080,
        "env_name": "cartpole",
        "exp_name": "value",
        "param_name": "trusted-value",
        "wandb_tags": ["trusted value"],
        "algo": "trusted-value",
        "num_timesteps": int(65000),
        "use_wandb": True,
        "val_epoch": 1000,
        "mini_batch_size": 2048,
        "n_val_envs": 16,
        "n_envs": 1024 + 16,
        "n_steps": 256,
        "n_pos_states": 10,
        "td_lmbda": False,
        #TODO: have fewer hidden dims?
        "hidden_dims": [256, 256, 256, 256],
        "save_pics_ascender": False,
    }
    run_next_hyperparameters(hparams)
