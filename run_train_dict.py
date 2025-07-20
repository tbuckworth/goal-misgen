from helper_local import get_model_with_largest_checkpoint
from hyperparameter_optimization import run_next_hyperparameters

if __name__ == '__main__':
    # # coinrun/maze:
    # hparams = {
    #     "num_levels": 100000,
    #     "architecture": "impala",
    #     "distribution_mode": "hard",
    #     "seed": 1080,
    #     "env_name": "maze_aisc",
    #     "exp_name": "value",
    #     "param_name": "trusted-value",
    #     "wandb_tags": ["trusted value"],
    #     "use_wandb": True,
    #     "val_epoch": 1000,
    #     "mini_batch_size": 2048,
    #     "n_val_envs": 4,
    #     "n_envs": 16 + 4,#shouldn't we use longer n_steps?
    #     "n_steps": 4096,
    #     "trusted_policy": "uniform",
    #     "update_frequently": True,
    #     # "trusted_temp": 7,
    #     "td_lmbda": False,
    # }
    # # # #cartpole:
    # hparams = {
    #     # "num_levels": 100000,
    #     "architecture": "mlpmodel",
    #     # "distribution_mode": "hard",
    #     "seed": 1080,
    #     "env_name": "mountain_car",
    #     "exp_name": "value",
    #     "param_name": "trusted-value",
    #     "wandb_tags": ["trusted value"],
    #     "algo": "trusted-value",
    #     "num_timesteps": int((128+16)*1024),
    #     "use_wandb": True,
    #     "val_epoch": 1000,
    #     "mini_batch_size": 2048,
    #     "n_val_envs": 16,
    #     "n_envs": 128 + 16,
    #     "n_steps": 4096,
    #     "n_pos_states": 10,
    #     "td_lmbda": False,
    #     # TODO: have fewer hidden dims?
    #     "hidden_dims": [256, 256],
    #     "save_pics_ascender": False,
    #     "update_frequently": True,
    #     "trusted_policy": "uniform",
    #     "trusted_temp": 7,
    # }
    # run_next_hyperparameters(hparams)
    # # Ascent:
    # hparams = {
    #     # "num_levels": 100000,
    #     "architecture": "mlpmodel",
    #     # "distribution_mode": "hard",
    #     "seed": 1080,
    #     "env_name": "acrobot",
    #     "exp_name": "value",
    #     "param_name": "trusted-value",
    #     "wandb_tags": ["trusted value"],
    #     "algo": "trusted-value",
    #     "num_timesteps": int((128+16)*1024),
    #     "use_wandb": True,
    #     "val_epoch": 1000,
    #     "mini_batch_size": 2048,
    #     "n_val_envs": 16,
    #     "n_envs": 128 + 16,
    #     "n_steps": 4096,
    #     "n_pos_states": 10,
    #     "td_lmbda": False,
    #     "hidden_dims": [256, 256, 256, 256],
    #     "save_pics_ascender": False,
    #     "trusted_policy": "uniform",
    #     "trusted_temp": 7,
    # }
    hparams = {
        "algo": "latent-diffusion",
        "exp_name": "diffusion",
        "env_name": "coinrun",
        "param_name": "latent-diffusion",
        "num_timesteps": 200000000,
        "num_checkpoints": 5,
        "seed": 6033,
        "epoch": 10,
        "num_levels": 1000,
        "distribution_mode": "hard",
        "use_wandb": True,
        "model_file": get_model_with_largest_checkpoint("logs/train/coinrun/coinrun/2025-01-24__15-27-41__seed_6033"),
    }
    load_and_run_maze = {
        "exp_name": "maze1",
        "env_name": "maze_aisc",
        "num_levels": 100000,
        "distribution_mode": "hard",
        "param_name": "hard-500-low-mem",
        "num_timesteps": 200000000,
        "num_checkpoints": 5,
        "seed": 50,
        "rand_region": 1,
        "use_wandb": True,
        "model_file": get_model_with_largest_checkpoint("logs/train/maze_aisc/maze1/2025-04-01__18-09-01__seed_50"),
        "render": True,
        "normalize_rew": False,
    }
    run_next_hyperparameters(load_and_run_maze)
