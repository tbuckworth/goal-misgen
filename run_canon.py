import re

from helper_local import get_model_with_largest_checkpoint
from hyperparameter_optimization import run_next_hyperparameters
from load_wandb_table import load_summary

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

unique_ascent_dirs = [
    'logs/train/ascent/ascent/2024-11-19__11-58-01__seed_1080',
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
    'logs/train/ascent/Ascent/2024-11-19__13-27-49__seed_81',
]

maze_dict = {
    "shifted": {
        "model_file": "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
        "exp_name": "maze1",
        "env_name": "maze_aisc",
        "num_levels": 100000,
        "distribution_mode": "hard",
        "param_name": "hard-500",
        "seed": 1080,
        "rand_region": 10,
    },
    "unshifted": {
        "model_file": "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",
        "exp_name": "maze1",
        "env_name": "maze_aisc",
        "num_levels": 100000,
        "distribution_mode": "hard",
        "param_name": "hard-500",
        "seed": 1080,
        "rand_region": 0,
    }
}

maze_dirs = [
    "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",#problem Unshifted rand.region = 0
    "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
]

new_maze_dirs = [
    "logs/train/maze_aisc/maze1/2024-11-25__15-28-05__seed_42/model_200015872.pth",  # rand.region = 10
    "logs/train/maze_aisc/maze1/2024-11-25__15-29-49__seed_42/model_200015872.pth",  # rand.region = 0 #problem
]

coinrun_dirs = [
    "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth", # random_percent = 0
    "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth", # random_percent = 10
    "logs/train/coinrun/coinrun/2025-01-22__09-43-00__seed_6033", # random_percent = 0, levels = 500
]

# generalising_ascender = [
#     get_model_with_largest_checkpoint('logs/train/ascent/Ascent/2024-11-19__12-04-10__seed_50')
# ]

ascent_misgeneralising_but_low_valid_distance = [
    "logs/train/ascent/Ascent/2024-11-19__12-55-04__seed_42"
]

maze_value_networks = {
    "Maze Value Original - fixed": "logs/train/maze_aisc/value/2024-11-23__10-38-36__seed_1080",
    # "Maze Value Dodgy": "logs/train/maze_aisc/value/2025-01-17__12-20-25__seed_1080",
    "Maze Value TD_Lambda": "logs/train/maze_aisc/value/2025-01-22__10-09-39__seed_1080",
    "Maze Unlimited TD_0 10 Epochs": "logs/train/maze_aisc/value/2025-01-22__17-40-33__seed_1080",
    "Maze Unlimited TD_0 200 Epochs": "logs/train/maze_aisc/value/2025-01-22__10-18-06__seed_1080",
}




def get_seed(model_file):
    try:
        return re.search(r"seed_(\d+)", model_file).group(1)
    except Exception as e:
        print("seed not found, using 42")
        return 42

def hp_run(model_file, tag_dict, tag):
    seed = get_seed(model_file)
    hparams = {
        # "model_file": model_file,
        "model_file": get_model_with_largest_checkpoint(model_file),
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
        "exp_name": "canon",
        "param_name": "ascent-canon",
        "wandb_tags": [tag],  # "pre-trained-value"],  # "coinrun misgen3"],
        "num_checkpoints": 1,
        "use_wandb": True,
        "num_timesteps": int(65000),
        "val_epoch": 100,
        "mini_batch_size": 2048,
        "n_val_envs": 32,
        "n_envs": int(256 + 32),
        "num_levels": 10000,
        "distribution_mode": "hard",
        "seed": seed,
        "hidden_dims": [256, 256, 256],

        "load_value_models": True,
        "value_dir": tag_dict[tag],
        "soft_canonicalisation": False,
        "meg": False,
        "remove_duplicate_actions": True,
        # "use_unique_obs": True,
        # "architecture": "crafted-policy",
        # "misgen": model_file,
        # "learning_rate": 1e-3,
    }
    run_next_hyperparameters(hparams)

def run_tags_for_files(tag_dict, model_files, ignore_errors=True):
    for tag in tag_dict.keys():
        for model_file in model_files:
            if not ignore_errors:
                hp_run(model_file, tag_dict, tag)
            else:
                try:
                    hp_run(model_file, tag_dict, tag)
                except Exception as e:
                    print(e)
                    try:
                        import wandb
                        wandb.finish()
                    except Exception as e:
                        pass
        try:
            load_summary(env=tag, exclude_crafted=True, tag=tag)
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':
    # tag = "canon maze hard grouped actions tdlmbda"
    model_files = maze_dirs + new_maze_dirs
    # run_tags_for_files({"Maze Value Original - fixed1": "logs/train/maze_aisc/value/2024-11-23__10-38-36__seed_1080"},
    #                    model_files, ignore_errors=False)

    model_files = unique_ascent_dirs
    run_tags_for_files({"Ascent_Hard_Canon_corrected":None}, model_files, ignore_errors=False)
