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
                    filters={"$and": [{"tags": "misgen-canon", "state": "finished"}]}
                    )

    # Collect and filter data
    filtered_data = []
    for run in runs:
        table = run.use_artifact("distances:latest").get("distances")
        for row in table.data:
            x, y, group = row
            # if x > 2 and y < 5:  # Example filter condition
            filtered_data.append([x, y, group])



if __name__ == "__main__":
    load()
