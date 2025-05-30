import os

from helper_local import get_rew_term
try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

def get_performing_model_dirs(env_name):
    target_rew = get_rew_term(env_name)
    wandb_login()
    api     = wandb.Api()
    entity, project = "ic-ai-safety", "goal-misgen"

    # include summary.mean_episode_rewards ≥ target_rew in the filter
    filters = {
        "state":                 "finished",
        "config.algo":           "ppo",
        "config.env_name":       env_name,
        "summary_metrics.mean_episode_rewards": {"$gte": target_rew}
    }

    runs = api.runs(f"{entity}/{project}", filters=filters)
    model_dirs = []
    for run in runs:
        logdir = run.config.get("logdir")
        if logdir and os.path.exists(logdir):
            model_dirs.append(logdir)

    if not model_dirs:
        raise FileNotFoundError("No local models found for this env. Could have been trained elsewhere.")
    return model_dirs

def get_performing_model_dirs(env_name):
    target_rew = get_rew_term(env_name)
    # do a wandb search for models over this reward, algo = ppo, state = finished, return logdirs
    wandb_login()
    api = wandb.Api()
    entity, project = "ic-ai-safety", "goal-misgen"
    filters = {"$and": [{"state": "finished",
                         "config.algo": "ppo",
                         "config.env_name": env_name,
                         "summary_metrics.mean_episode_rewards": {"$gte": target_rew}
                         }]}
    runs = api.runs(entity + "/" + project,
                    filters=filters
                    )
    summary_list, config_list, name_list, state_list = [], [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    model_dirs = []
    for s, c, n in zip(summary_list, config_list, name_list):
        if s["mean_episode_rewards"] >= target_rew:
            model_dirs.append(c.get("logdir",None))
    # filter model_dirs for dirs that exist using os.path.exists:
    local_dirs = [d for d in model_dirs if os.path.exists(d)]
    if len(local_dirs) == 0:
        raise FileNotFoundError("No local models found for this env. Could have been trained elsewhere.")
    return local_dirs

def main(env_name):
    model_dirs = get_performing_model_dirs(env_name)




if __name__ == "__main__":
    main("mountain_car")
