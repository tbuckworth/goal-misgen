import pandas as pd
from matplotlib import pyplot as plt

try:
    import wandb
    from private_login import wandb_login
except ImportError:
    pass


def graph_wandb_res():
    data = pd.read_csv("data/dist_metrics.csv")

    id_vars = ['Norm', 'Metric', 'mean_episode_rewards', 'val_mean_episode_rewards', 'run']
    df2 = data.pivot(columns='Env', values='Distance', index=id_vars).reset_index()
    df2["Diff"] = df2["Valid"] - df2["Train"]
    df = df2.melt(
        id_vars =id_vars,
        value_vars = ["Train","Valid","Diff"],
        var_name = "Env",
        value_name = "Distance",
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


    correls = df2.groupby(['Norm', 'Metric'])[['val_mean_episode_rewards','Diff']].corr().iloc[0::2,-1].reset_index()
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


if __name__ == "__main__":
    load()
