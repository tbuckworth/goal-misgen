
from hyperparameter_optimization import run_next_hyperparameters

if __name__ == '__main__':
    hparams = {
        "model_file": "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
        "epoch": 0,
        "algo": "canon",
        "env_name": "get",
        "exp_name": "maze1",
        "wandb_tags": ["maze1 canon"],
        "num_checkpoints": 1,
        "use_wandb": True,
        "num_timesteps": int(65000),
        "val_epoch": 200,
        # "learning_rate": 1e-3,
    }
    run_next_hyperparameters(hparams)
