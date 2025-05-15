from helper_local import get_model_with_largest_checkpoint, get_seed
from hyperparameter_optimization import run_next_hyperparameters
from load_wandb_table import load_summary, load_meg
from concurrent.futures import ThreadPoolExecutor, as_completed

ascent_represent = [
    # misgen:
    "logs/train/ascent/Ascent/2024-11-19__12-10-48__seed_81",
    # goal misgen:
    "logs/train/ascent/Ascent/2024-11-19__12-16-12__seed_42",
    # gen:
    "logs/train/ascent/Ascent/2024-11-19__12-37-50__seed_6033",
]

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
    "logs/train/maze_aisc/maze1/2024-11-08__15-54-16__seed_1080/model_200015872.pth",
    # problem Unshifted rand.region = 0
    "logs/train/maze_aisc/maze1/2024-11-11__20-51-51__seed_1080/model_200015872.pth",
]

new_maze_dirs = [
    "logs/train/maze_aisc/maze1/2024-11-25__15-28-05__seed_42/model_200015872.pth",  # rand.region = 10
    "logs/train/maze_aisc/maze1/2024-11-25__15-29-49__seed_42/model_200015872.pth",  # rand.region = 0 #problem
]

maze_dirs_apr25 = [
    # already run - put back in:
    "logs/train/maze_aisc/maze1/2025-04-01__18-09-01__seed_50/model_80019456.pth",  # rand.region = 1
    "logs/train/maze/maze1/2025-04-03__12-27-33__seed_118/model_120061952.pth", # rand.region = 0
    "logs/train/maze_aisc/maze1/2025-04-03__18-22-05__seed_51/model_80019456.pth", # rand.region = 3
    "logs/train/maze_aisc/maze1/2025-04-04__07-43-53__seed_1993", # rand.region = 10
]

maze_test = [
    "logs/train/maze_aisc/maze1/2025-04-03__18-22-05__seed_51/model_80019456.pth"
]

coinrun_dirs = [
    "logs/train/coinrun/coinrun/2024-10-05__17-20-34__seed_6033/model_200015872.pth",  # random_percent = 0
    "logs/train/coinrun/coinrun/2024-10-05__18-06-44__seed_6033/model_200015872.pth",  # random_percent = 10
    # "logs/train/coinrun/coinrun/2025-01-22__09-43-00__seed_6033", # random_percent = 0, levels = 500
    "logs/train/coinrun/coinrun/2025-01-24__15-27-41__seed_6033",  # rp = 0, levels = 1000 (still running, so uncomment)
    "logs/train/coinrun/coinrun/2025-01-24__15-30-53__seed_6033",  # rp = 0, levels = 2000 (still running, uncomment)
]

new_coinrun_dirs = [
    "logs/train/coinrun/coinrun/2025-01-28__11-36-54__seed_42", # rand_percent = 0
    "logs/train/coinrun/coinrun/2025-01-28__11-36-54__seed_0", # rand_percent = 0
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

cartpole_dirs = [
    "logs/train/cartpole/cartpole/2025-01-30__12-19-17__seed_42",
    "logs/train/cartpole/cartpole/2025-01-30__12-15-05__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__11-41-35__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__09-33-34__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__09-00-27__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__08-58-21__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__08-38-19__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__08-29-54__seed_50",
    "logs/train/cartpole/cartpole/2025-01-30__08-28-16__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__08-07-36__seed_50",
    "logs/train/cartpole/cartpole/2025-01-30__08-01-11__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__07-19-57__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__07-05-24__seed_42",
    "logs/train/cartpole/cartpole/2025-01-30__06-54-50__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__06-36-55__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__05-49-26__seed_50",
    "logs/train/cartpole/cartpole/2025-01-30__05-30-11__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__04-56-24__seed_0",
    "logs/train/cartpole/cartpole/2025-01-30__04-48-34__seed_0",
    "logs/train/cartpole/cartpole/2025-01-30__04-29-50__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__04-17-35__seed_0",
    "logs/train/cartpole/cartpole/2025-01-30__04-15-10__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__04-03-05__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__03-26-46__seed_42",
    "logs/train/cartpole/cartpole/2025-01-30__03-20-35__seed_0",
    "logs/train/cartpole/cartpole/2025-01-30__03-11-18__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__02-56-56__seed_50",
    "logs/train/cartpole/cartpole/2025-01-30__02-45-39__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__02-44-18__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__01-58-59__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__01-50-33__seed_81",
    "logs/train/cartpole/cartpole/2025-01-30__01-07-57__seed_50",
    "logs/train/cartpole/cartpole/2025-01-30__01-07-24__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-30__00-37-31__seed_0",
    "logs/train/cartpole/cartpole/2025-01-30__00-04-35__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__23-44-01__seed_42",
    "logs/train/cartpole/cartpole/2025-01-29__23-01-55__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__22-30-38__seed_0",
    "logs/train/cartpole/cartpole/2025-01-29__22-19-22__seed_0",
    "logs/train/cartpole/cartpole/2025-01-29__20-51-44__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__20-38-01__seed_0",
    "logs/train/cartpole/cartpole/2025-01-29__20-35-19__seed_42",
    "logs/train/cartpole/cartpole/2025-01-29__20-34-19__seed_50",
    "logs/train/cartpole/cartpole/2025-01-29__20-32-57__seed_0",
    "logs/train/cartpole/cartpole/2025-01-29__20-10-36__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__19-48-55__seed_42",
    "logs/train/cartpole/cartpole/2025-01-29__19-28-37__seed_0",
    "logs/train/cartpole/cartpole/2025-01-29__18-47-15__seed_81",
    "logs/train/cartpole/cartpole/2025-01-29__18-17-15__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__18-15-47__seed_6033",
    "logs/train/cartpole/cartpole/2025-01-29__17-35-24__seed_81",
    "logs/train/cartpole/cartpole/2025-01-29__16-19-28__seed_42",
    "logs/train/cartpole/cartpole/2025-01-29__15-59-42__seed_81",
    "logs/train/cartpole/cartpole/2025-01-29__15-59-04__seed_50",
    "logs/train/cartpole/cartpole/2025-01-29__11-12-26__seed_6033",
]

cartpole_value_networks = {
    "Cartpole 400": "logs/train/cartpole/value/2025-01-29__11-04-45__seed_1080",
}


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
        "param_name": "cartpole-canon",
        "wandb_tags": [tag],  # "pre-trained-value"],  # "coinrun misgen3"],
        "num_checkpoints": 1,
        "use_wandb": True,
        "num_timesteps": int(4096*24),
        "val_epoch": 25,
        "mini_batch_size": 2048,
        "n_val_envs": 8,
        "n_envs": int(16 + 8),
        "n_steps": 4096,
        "num_levels": 10000,
        "learning_rate": 5e-4,
        "distribution_mode": "hard",
        "seed": seed,
        "hidden_dims": [256, 256, 256, 256],

        "load_value_models": True,
        "pre_trained_value_encoder": True,
        "value_dir": tag_dict[tag],
        "soft_canonicalisation": True,
        "update_frequently": True,
        "infinite_value": True,
        "meg": False,
        "remove_duplicate_actions": True,
        "centered_logprobs": False,
        "adjust_logprob_mean": False,
        "use_valid_env": True,
        "meg_version": "kldiv",
        "pirc": True,
        "trusted_policy": "uniform",
        "trusted_temp": 7,
        # "subject_temp": 5,
        "meg_ground_next": True,
        "consistency_coef": 10.,
        "use_min_val_loss": False,
    }
    run_next_hyperparameters(hparams)


def run_tags_for_files(tag_dict, model_files, ignore_errors=True):
    if not isinstance(model_files, list):
        model_files = model_files.split(" ")
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
            load_meg([tag])
            # load_summary(env=tag, exclude_crafted=True, tag=tag)
        except Exception as e:

            print(e)
            pass


def run_tags_for_files_threaded(tag_dict, model_files, ignore_errors=True):
    def safe_hp_run(model_file, tag_dict, tag):
        """
        Wrapper that catches exceptions if ignore_errors is True.
        """
        try:
            hp_run(model_file, tag_dict, tag)
        except Exception as e:
            print(f"Error running hp_run on {model_file} with tag {tag}: {e}")
            # Attempt to finish wandb run if something goes wrong
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    for tag in tag_dict.keys():
        # Create a thread pool with 10 workers
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            # Submit one task per model_file
            for model_file in model_files:
                if not ignore_errors:
                    # If we're NOT ignoring errors, no need for a wrapper
                    futures.append(executor.submit(hp_run, model_file, tag_dict, tag))
                else:
                    # If we ARE ignoring errors, use safe_hp_run
                    futures.append(executor.submit(safe_hp_run, model_file, tag_dict, tag))

            # Optionally wait for all futures to complete and handle results
            for future in as_completed(futures):
                # If you do want to see raised exceptions in ignore_errors=False scenario:
                try:
                    future.result()
                except Exception as e:
                    print(f"Unhandled exception: {e}")

        # Once all model_files have been processed for this tag, load summary
        try:
            load_summary(env=tag, exclude_crafted=True, tag=tag)
        except Exception as e:
            print(f"Error in load_summary with tag {tag}: {e}")


if __name__ == '__main__':
    # run_tags_for_files({"Test": None}, cartpole_dirs, ignore_errors=False)

    # run_tags_for_files({"Cartpole_Meg_KL0": None}, cartpole_dirs, ignore_errors=True)

    # # model_files = maze_dirs + new_maze_dirs
    run_tags_for_files({"Maze_VOrig_Soft_Inf_Pre": None}, maze_dirs_apr25, ignore_errors=True)
    # run_tags_for_files({"Maze_VOrig_Soft_Target_Mean_Adj": None}, maze_dirs , ignore_errors=True)
    # run_tags_for_files({"Maze_VOrig_Soft_Target_Mean_Adj": None}, new_maze_dirs, ignore_errors=True)
    # run_tags_for_files({"Maze_VOrig_Soft_Target_Mean_Adj": None}, maze_dirs_apr25, ignore_errors=True)
    # run_tags_for_files({"Maze_VOrig_Soft_Target_Mean_Adj": None}, maze_dirs + maze_dirs_apr25, ignore_errors=True)

    # run_tags_for_files({"test uniform target infinite": None}, maze_test, ignore_errors=True)
    # run_tags_for_files({"test ascent":None}, local_unique_ascent_dirs[:1], ignore_errors=False)


    #
    # run_tags_for_files({"new ascent uniform no inf":None}, local_unique_ascent_dirs, ignore_errors=False)
    # run_tags_for_files({"new cartpole tempered target mean":None}, cartpole_dirs, ignore_errors=True)
    # run_tags_for_files({"new cartpole target mean old val":None}, cartpole_dirs, ignore_errors=True)
    # run_tags_for_files({"new ascent target mean":None}, unique_ascent_dirs, ignore_errors=True)

    # run_tags_for_files({"test":None}, unique_ascent_dirs, ignore_errors=False)
    # #
    # run_tags_for_files({"Coinrun_Soft_Inf": None}, new_coinrun_dirs, ignore_errors=True)
    # run_tags_for_files({"new coinrun double tempered": None}, coinrun_dirs + new_coinrun_dirs, ignore_errors=True)
