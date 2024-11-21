from hyperparameter_optimization import run_next_hyperparameters

local_unique_ascent_dirs = [
    "logs/train/ascent/Ascent/2024-11-19__12-41-35__seed_6033",
    "logs/train/ascent/Ascent/2024-11-19__13-22-57__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__12-38-08__seed_6033",
    "logs/train/ascent/Ascent/2024-11-19__13-06-24__seed_0",
    "logs/train/ascent/Ascent/2024-11-19__12-58-38__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__12-33-33__seed_0",
    "logs/train/ascent/Ascent/2024-11-19__12-43-59__seed_0",
    "logs/train/ascent/Ascent/2024-11-19__13-22-57__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__12-58-38__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__13-22-57__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__12-37-50__seed_6033",
    "logs/train/ascent/Ascent/2024-11-19__12-16-12__seed_42",
    "logs/train/ascent/Ascent/2024-11-19__13-27-49__seed_81",
    "logs/train/ascent/Ascent/2024-11-19__12-10-48__seed_81",
    "logs/train/ascent/Ascent/2024-11-19__12-33-02__seed_42",
    "logs/train/ascent/Ascent/2024-11-19__12-43-07__seed_50",
    "logs/train/ascent/Ascent/2024-11-19__13-10-25__seed_42",
    "logs/train/ascent/Ascent/2024-11-19__12-16-12__seed_42",
    "logs/train/ascent/Ascent/2024-11-19__12-33-33__seed_0",
    "logs/train/ascent/Ascent/2024-11-19__12-17-49__seed_81",
    "logs/train/ascent/Ascent/2024-11-19__12-33-02__seed_42",
    "logs/train/ascent/Ascent/2024-11-19__13-27-49__seed_81",
]

unique_ascent_dirs = ['logs/train/ascent/ascent/2024-11-19__11-58-01__seed_1080',
                      'logs/train/ascent/ascent/2024-11-19__11-58-34__seed_1080',
                      'logs/train/ascent/Ascent/2024-11-19__12-04-10__seed_50',
                      'logs/train/ascent/Ascent/2024-11-19__12-04-27__seed_6033',
                      'logs/train/ascent/Ascent/2024-11-19__12-04-32__seed_42',
                      'logs/train/ascent/Ascent/2024-11-19__12-04-40__seed_0',
                      'logs/train/ascent/Ascent/2024-11-19__12-10-48__seed_81',
                      'logs/train/ascent/Ascent/2024-11-19__12-17-49__seed_81',
                      'logs/train/ascent/Ascent/2024-11-19__12-20-19__seed_0',
                      'logs/train/ascent/Ascent/2024-11-19__12-33-33__seed_0',
                      'logs/train/ascent/Ascent/2024-11-19__12-37-43__seed_50',
                      'logs/train/ascent/Ascent/2024-11-19__12-37-43__seed_81',
                      'logs/train/ascent/Ascent/2024-11-19__12-37-50__seed_6033',
                      'logs/train/ascent/Ascent/2024-11-19__12-38-08__seed_6033',
                      'logs/train/ascent/Ascent/2024-11-19__12-38-59__seed_42',
                      'logs/train/ascent/Ascent/2024-11-19__12-39-01__seed_6033',
                      'logs/train/ascent/Ascent/2024-11-19__12-41-35__seed_6033',
                      'logs/train/ascent/Ascent/2024-11-19__12-43-07__seed_50',
                      'logs/train/ascent/Ascent/2024-11-19__12-43-59__seed_0',
                      'logs/train/ascent/Ascent/2024-11-19__12-55-04__seed_42',
                      'logs/train/ascent/Ascent/2024-11-19__12-55-21__seed_81',
                      'logs/train/ascent/Ascent/2024-11-19__13-06-24__seed_0',
                      'logs/train/ascent/Ascent/2024-11-19__13-10-25__seed_42',
                      'logs/train/ascent/Ascent/2024-11-19__13-27-49__seed_81']

maze_dirs = [
    "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",
    "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
]

coinrun_dirs = [
    "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth",
    "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",
]

if __name__ == '__main__':
    for model_file in local_unique_ascent_dirs:
        hparams = {
            "model_file": model_file,
            # coinrun unshifted
            # "model_file":  "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",

            # coinrun shifted
            # "model_file":  "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth",

            # unshifted maze:
            # "model_file":  "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",

            # shifted maze:
            # "model_file": "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
            "epoch": 0,
            "algo": "canon",
            "env_name": "get",
            "exp_name": "ascent",
            "param_name": "ascent-canon",
            "wandb_tags": ["canon misgen3"],
            "num_checkpoints": 1,
            "use_wandb": True,
            "num_timesteps": int(65000),
            "val_epoch": 200,
            "mini_batch_size": 2048,
            "n_val_envs": 128,
            "n_envs": 256 + 128,
            # "learning_rate": 1e-3,
        }
        run_next_hyperparameters(hparams)
