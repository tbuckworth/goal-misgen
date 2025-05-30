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


def main(env_name):
    model_dirs = get_performing_model_dirs(env_name)




if __name__ == "__main__":
    main("mountain_car")
