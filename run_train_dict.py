from hyperparameter_optimization import run_next_hyperparameters

if __name__ == '__main__':
    hparams = {
        # "num_levels": 100000,
        "architecture": "impala",
        # "distribution_mode": "hard",
        "seed": 1080,
        "env_name": "maze_aisc",
        "exp_name": "value",
        "param_name": "trusted-value",
        "wandb_tags": ["trusted value"],
        "algo": "trusted-value-unlimited",
        "use_wandb": True,
        "val_epoch": 200,
        "mini_batch_size": 2048,
        "n_val_envs": 16,
        "n_envs": 256 + 16,
        "n_steps": 256,
        "n_pos_states": 10,
        "td_lmbda": True,
        #TODO: have fewer hidden dims?
        "hidden_dims": [256, 256, 256, 256],
        "save_pics_ascender": False,
    }
    run_next_hyperparameters(hparams)
