import pandas as pd
from matplotlib import pyplot as plt

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass

local_run_names = [
    "breezy-sun-1179",
    "bright-sound-1216",
    "celestial-butterfly-1211",
    "comfy-sky-1176",
    "confused-dragon-1210",
    "copper-oath-1175",
    "Crafted Generaliser",
    "Crafted Misgeneraliser",
    "northern-blaze-1250",
    "dainty-galaxy-1191",
    "celestial-butterfly-1211",
    "copper-oath-1175",
    "celestial-butterfly-1211",
    "deft-puddle-1177",
    "electric-wood-1278",
    "dulcet-snow-1215",
    "fine-aardvark-1230",
    "giddy-star-1232",
    "comfy-sky-1176",
    "deft-puddle-1177",
    "northern-blaze-1250",
    "dulcet-snow-1215",
    "electric-wood-1278",
]

run_names = [
    "absurd-field-1219",
    "breezy-sun-1179",
    "bright-sound-1216",
    "celestial-butterfly-1211",
    "chocolate-galaxy-1214",
    "classic-wood-1246",
    "comfy-sky-1176",
    "confused-dragon-1210",
    "copper-oath-1175",
    "Crafted Generaliser",
    "Crafted Misgeneraliser",
    "crimson-frost-1208",
    "crisp-violet-1221",
    "dainty-galaxy-1191",
    "celestial-butterfly-1211",
    "copper-oath-1175",
    "celestial-butterfly-1211",
    "deep-valley-1212",
    "deft-puddle-1177",
    "devoted-frog-1269",
    "driven-surf-1185",
    "dulcet-snow-1215",
    "fine-aardvark-1230",
    "frosty-valley-1220",
    "giddy-star-1232",
    "glad-flower-1249",
    "comfy-sky-1176",
    "deft-puddle-1177",
    "crimson-frost-1208",
    "pleasant-bee-1190",
    "dulcet-snow-1215",
    "devoted-frog-1269",
]


def graph_wandb_res():
    data = pd.read_csv("data/dist_metrics.csv")

    id_vars = ['Norm', 'Metric', 'mean_episode_rewards', 'val_mean_episode_rewards', 'run']
    df2 = data.pivot(columns='Env', values='Distance', index=id_vars).reset_index()
    df2["Diff"] = df2["Valid"] - df2["Train"]
    df = df2.melt(
        id_vars=id_vars,
        value_vars=["Train", "Valid", "Diff"],
        var_name="Env",
        value_name="Distance",
    )

    # Correcting the legend addition logic to handle it per axis instead of plt level
    # Adjusting the code to ensure proper legend handling and no overlap issues
    unique_combinations = df[['Norm', 'Metric']].drop_duplicates()
    fig, axes = plt.subplots(
        nrows=len(df['Norm'].unique()),
        ncols=len(df['Metric'].unique()),
        figsize=(15, 10),
        sharex=True,
        sharey=False
    )

    axes = axes.flatten()

    for idx, (norm, metric) in enumerate(unique_combinations.itertuples(index=False)):
        ax = axes[idx]
        subset = df[(df['Norm'] == norm) & (df['Metric'] == metric)]

        scatter = ax.scatter(
            subset['val_mean_episode_rewards'],
            subset['Distance'],
            c=subset['Env'].astype('category').cat.codes,
            cmap='viridis',
            alpha=0.7
        )

        ax.set_title(f'Norm: {norm}, Metric: {metric}')
        ax.set_xlabel('val_mean_episode_rewards')
        ax.set_ylabel('Distance')

        if idx == 0:  # Add legend to only the first subplot for simplicity
            env_categories = subset['Env'].astype('category').cat.categories
            color_legend = [
                plt.Line2D(
                    [0], [0], marker='o',
                    color='w', markerfacecolor=plt.cm.viridis(i / len(env_categories)),
                    markersize=10)
                for i in range(len(env_categories))]
            ax.legend(color_legend, env_categories, title="Env", loc='upper right')

    plt.tight_layout()
    plt.show()

    correls = df2.groupby(['Norm', 'Metric'])[['val_mean_episode_rewards', 'Diff']].corr().iloc[0::2, -1].reset_index()
    correls = correls.drop(columns=["Env"])
    correls.rename(columns={'Diff': 'Correlation'}, inplace=True)


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


def loadtemp():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen2", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = [run.config["logdir"] for run in runs if run.name in local_run_names]
    for run in runs:
        if run.name in run_names:
            print(run.config["logdir"])


def load_all():
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
            df["logdir"] = run.config["logdir"]
            df["user"] = run.user

            all_data.append(df)

    final = pd.concat(all_data).reset_index(drop=True)
    final.to_csv("data/dist_metrics_full.csv", index=False)
    print(final)


def load_summary():
    # Fetch runs from a project
    api = wandb.Api()
    project_name = "goal-misgen"
    runs = api.runs(f"ic-ai-safety/{project_name}",
                    filters={"$and": [{"tags": "canon misgen3", "state": "finished"}]}
                    )

    # Collect and filter data
    all_data = []
    for run in runs:
        row = {}
        row["Mean Training Episode Rewards"] = run.summary.mean_episode_rewards
        row["Mean Validation Episode Rewards"] = run.summary.val_mean_episode_rewards
        row["Training Distance"] = run.summary.l2_normalized_l2_distance_Training
        row["Validation Distance"] = run.summary.l2_normalized_l2_distance_Validation
        row["run"] = run.name
        row["logdir"] = run.config["logdir"]
        all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv("data/l2_dist.csv", index=False)
    print(df)


if __name__ == "__main__":
    load_summary()
