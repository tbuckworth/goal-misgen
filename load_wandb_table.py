import pandas as pd

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass


def load():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen2", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = []
    for run in runs:
        # table = run.use_artifact("distances:latest").get("distances")
        table = [x for x in run.logged_artifacts()][0].get('distances')
        if table is not None:
            df = pd.DataFrame(data=table.data, columns=table.columns)
            df["mean_episode_rewards"] = run.summary.mean_episode_rewards
            df["val_mean_episode_rewards"] = run.summary.val_mean_episode_rewards
            df["run"] = run.name
            all_data.append(df)

    final = pd.concat(all_data).reset_index(drop=True)
    final.to_csv("data/dist_metrics.csv", index=False)

if __name__ == "__main__":
    load()
