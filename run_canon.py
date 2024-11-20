
from hyperparameter_optimization import run_next_hyperparameters

if __name__ == '__main__':
    for model_file in [
        # "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",
        "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth",
        "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth"
    ]:
        hparams = {
            "model_file": model_file,
            # coinrun unshifted
            # "model_file":  "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",

            # coinrun shifted
            # "model_file":  "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth",

            #unshifted maze:
            # "model_file":  "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",

            #shifted maze:
            # "model_file": "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
            "epoch": 0,
            "algo": "canon",
            "env_name": "get",
            "exp_name": "coinrun",
            "param_name": "ascent-canon",
            "wandb_tags": ["coinrun canon"],
            "num_checkpoints": 1,
            "use_wandb": True,
            "num_timesteps": int(65000),
            "val_epoch": 1000,
            "mini_batch_size": 2048,
            # "learning_rate": 1e-3,
        }
        run_next_hyperparameters(hparams)
